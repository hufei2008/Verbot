#include "tts_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Python.h 必须第一个包含
#include <Python.h>

// ============================================================
// TtsEngine — Embedded Python TTS Engine
//
// 核心：使用 CPython 嵌入式 API 在 C++ 进程内运行 Python 代码
// - Py_Initialize() 启动解释器
// - cosyvoice_bridge.py 加载本地 TTS 模型
// - PyObject_CallFunction 调用 Python 函数
// - 通过 numpy C API 直接读取数组内存
// ============================================================

TtsEngine::TtsEngine() {}

TtsEngine::~TtsEngine() {
    cancel();

    // 等待工作线程退出
    if (m_worker.joinable()) {
        m_running = false;
        m_cv.notify_one();
        m_worker.join();
    }

    if (m_python_initialized && Py_IsInitialized()) {
        // MLX-Audio / Hugging Face can leave Python runtime resources alive.
        // Restore the main thread state and keep the GIL held for process
        // shutdown; otherwise Python/MLX atexit handlers can touch Python
        // state while the GIL is released.
        if (m_python_gil_released && m_python_main_thread_state) {
            PyEval_RestoreThread(static_cast<PyThreadState*>(m_python_main_thread_state));
            m_python_main_thread_state = nullptr;
        }
        m_py_bridge_module = nullptr;
        m_python_initialized = false;
        m_python_gil_released = false;
    } else if (m_py_bridge_module) {
        m_py_bridge_module = nullptr;
    }
}

// ============================================================
// 初始化
// ============================================================

bool TtsEngine::init(const std::string& cosyvoice_model_dir,
                     const std::string& python_home,
                     const std::string& bridge_script_dir) {
    if (m_initialized) return true;

    m_model_dir = cosyvoice_model_dir;
    m_python_home = python_home;
    m_bridge_script_dir = bridge_script_dir;

    fprintf(stdout, "[TtsEngine] Initializing with:\n");
    fprintf(stdout, "  model_dir:       %s\n", m_model_dir.c_str());
    fprintf(stdout, "  python_home:     %s\n", m_python_home.c_str());
    fprintf(stdout, "  bridge_script:   %s\n", m_bridge_script_dir.c_str());

    // 1. 初始化嵌入式 Python 解释器
    if (!ensure_python_initialized()) {
        fprintf(stderr, "[TtsEngine] Python initialization failed!\n");
        return false;
    }

    // 2. 加载 bridge 模块并初始化 TTS 模型
    if (!load_bridge_module()) {
        fprintf(stderr, "[TtsEngine] Load bridge module failed!\n");
        return false;
    }

    // 3. 调用 bridge.init() 加载模型
    {
        PyGILState_STATE gstate = PyGILState_Ensure();

        PyObject* bridge = static_cast<PyObject*>(m_py_bridge_module);
        PyObject* result = PyObject_CallMethod(bridge, "init", "(s)",
                                                m_model_dir.c_str());
        if (!result) {
            PyErr_Print();
            PyGILState_Release(gstate);
            fprintf(stderr, "[TtsEngine] bridge.init() failed!\n");
            return false;
        }
        Py_DECREF(result);
        PyGILState_Release(gstate);
    }

    // 4. 启动后台工作线程
    m_running = true;
    m_worker = std::thread(&TtsEngine::worker_loop, this);

    m_initialized = true;
    fprintf(stdout, "[TtsEngine] Successfully initialized!\n");
    return true;
}

// ============================================================
// Python 解释器初始化
// ============================================================

bool TtsEngine::ensure_python_initialized() {
    if (m_python_initialized) return true;

    // 如果 Python 已经初始化（例如被其他模块调用过），跳过
    if (Py_IsInitialized()) {
        m_python_initialized = true;
        return true;
    }

    // 设置 Python 环境路径，指向 TTS conda 环境
    // 这样 import 时能找到 numpy, mlx_audio 等
#ifdef Py_SetPythonHome
    Py_SetPythonHome(const_cast<char*>(m_python_home.c_str()));
#endif

    // 初始化解释器
    Py_Initialize();

    if (!Py_IsInitialized()) {
        fprintf(stderr, "[TtsEngine] Py_Initialize() failed!\n");
        return false;
    }

    // 添加 bridge 脚本路径到 sys.path
    PyRun_SimpleString(("import sys; sys.path.insert(0, '" +
                        m_bridge_script_dir + "')").c_str());

    std::string python_exe = m_python_home + "/bin/python";
    PyRun_SimpleString(
        ("import sys, multiprocessing\n"
         "sys.executable = '" + python_exe + "'\n"
         "multiprocessing.set_executable('" + python_exe + "')\n").c_str());

    // 验证关键模块可导入
    int ret = PyRun_SimpleString(
        "try:\n"
        "    import numpy\n"
        "    print(f'[TtsEngine] numpy {numpy.__version__} loaded')\n"
        "except Exception as e:\n"
        "    print(f'[TtsEngine] WARNING: numpy import failed: {e}')\n"
    );
    (void)ret;

    m_python_initialized = true;
    m_python_main_thread_state = PyEval_SaveThread();
    m_python_gil_released = true;
    fprintf(stdout, "[TtsEngine] Python interpreter initialized for local TTS\n");
    return true;
}

// ============================================================
// 加载 bridge 模块
// ============================================================

bool TtsEngine::load_bridge_module() {
    PyGILState_STATE gstate = PyGILState_Ensure();

    // 导入 cosyvoice_bridge 模块
    PyObject* module_name = PyUnicode_DecodeFSDefault("cosyvoice_bridge");
    PyObject* module = PyImport_Import(module_name);
    Py_DECREF(module_name);

    if (!module) {
        PyErr_Print();
        PyGILState_Release(gstate);
        fprintf(stderr, "[TtsEngine] Failed to import cosyvoice_bridge module!\n");
        return false;
    }

    m_py_bridge_module = static_cast<void*>(module);
    PyGILState_Release(gstate);

    fprintf(stdout, "[TtsEngine] cosyvoice_bridge module loaded\n");
    return true;
}

// ============================================================
// 同步合成
// ============================================================

bool TtsEngine::synthesize_sync(const std::string& text,
                                std::vector<int16_t>& out_pcm,
                                const std::string& spk_id) {
    if (!m_initialized) {
        fprintf(stderr, "[TtsEngine] Not initialized!\n");
        return false;
    }

    out_pcm.clear();

    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* bridge = static_cast<PyObject*>(m_py_bridge_module);

    // 调用 bridge.synthesize(text, spk_id, "sft")
    PyObject* result = PyObject_CallMethod(bridge, "synthesize", "(sss)",
                                           text.c_str(),
                                           spk_id.c_str(),
                                           "sft");
    if (!result) {
        PyErr_Print();
        PyGILState_Release(gstate);
        fprintf(stderr, "[TtsEngine] bridge.synthesize() failed!\n");
        return false;
    }

    // 通过 numpy.ndarray.tobytes() 获取数据（通用方式，不依赖 numpy 头文件细节）
    PyObject* buffer = PyObject_CallMethod(result, "tobytes", nullptr);
    if (!buffer) {
        PyErr_Print();
        Py_DECREF(result);
        PyGILState_Release(gstate);
        fprintf(stderr, "[TtsEngine] result.tobytes() call failed!\n");
        return false;
    }

    // 获取字节数据
    const char* buf_data = nullptr;
    Py_ssize_t buf_len = 0;
    if (PyBytes_AsStringAndSize(buffer, const_cast<char**>(&buf_data), &buf_len) == -1) {
        Py_DECREF(buffer);
        Py_DECREF(result);
        PyGILState_Release(gstate);
        fprintf(stderr, "[TtsEngine] PyBytes_AsStringAndSize failed!\n");
        return false;
    }

    // int16 = 2 bytes per sample
    size_t n_samples = buf_len / sizeof(int16_t);
    out_pcm.resize(n_samples);
    if (n_samples > 0) {
        memcpy(out_pcm.data(), buf_data, buf_len);
    }

    Py_DECREF(buffer);
    Py_DECREF(result);
    PyGILState_Release(gstate);

    if (!out_pcm.empty()) {
        int sample_rate = 24000;
        if (const char* env_sample_rate = std::getenv("TTS_SAMPLE_RATE")) {
            int configured_sample_rate = std::atoi(env_sample_rate);
            if (configured_sample_rate > 0) {
                sample_rate = configured_sample_rate;
            }
        }
        fprintf(stdout, "[TtsEngine] Synthesized %zu samples (%zu ms)\n",
                out_pcm.size(), out_pcm.size() * 1000 / (size_t)sample_rate);
    } else {
        fprintf(stderr, "[TtsEngine] Empty synthesis result!\n");
        return false;
    }

    return true;
}

bool TtsEngine::synthesize_stream(const std::string& text,
                                  const std::string& spk_id,
                                  TtsStreamChunkCallback on_chunk) {
    if (!m_initialized) {
        fprintf(stderr, "[TtsEngine] Not initialized!\n");
        return false;
    }
    if (!on_chunk) {
        fprintf(stderr, "[TtsEngine] Missing stream chunk callback!\n");
        return false;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* bridge = static_cast<PyObject*>(m_py_bridge_module);
    PyObject* generator = PyObject_CallMethod(bridge, "synthesize_stream", "(sss)",
                                              text.c_str(),
                                              spk_id.c_str(),
                                              "sft");
    if (!generator) {
        PyErr_Print();
        PyGILState_Release(gstate);
        fprintf(stderr, "[TtsEngine] bridge.synthesize_stream() failed!\n");
        return false;
    }

    PyObject* iterator = PyObject_GetIter(generator);
    Py_DECREF(generator);
    if (!iterator) {
        PyErr_Print();
        PyGILState_Release(gstate);
        fprintf(stderr, "[TtsEngine] synthesize_stream result is not iterable!\n");
        return false;
    }

    bool ok = true;
    size_t total_samples = 0;
    int chunk_count = 0;

    while (ok) {
        PyObject* chunk = PyIter_Next(iterator);
        if (!chunk) {
            if (PyErr_Occurred()) {
                PyErr_Print();
                ok = false;
            }
            break;
        }

        PyObject* buffer = PyObject_CallMethod(chunk, "tobytes", nullptr);
        if (!buffer) {
            PyErr_Print();
            Py_DECREF(chunk);
            ok = false;
            break;
        }

        const char* buf_data = nullptr;
        Py_ssize_t buf_len = 0;
        if (PyBytes_AsStringAndSize(buffer, const_cast<char**>(&buf_data), &buf_len) == -1) {
            Py_DECREF(buffer);
            Py_DECREF(chunk);
            ok = false;
            break;
        }

        std::vector<int16_t> pcm(buf_len / sizeof(int16_t));
        if (!pcm.empty()) {
            memcpy(pcm.data(), buf_data, pcm.size() * sizeof(int16_t));
            total_samples += pcm.size();
            chunk_count++;
        }

        Py_DECREF(buffer);
        Py_DECREF(chunk);

        if (!pcm.empty() && !on_chunk(pcm)) {
            ok = false;
            break;
        }
    }

    Py_DECREF(iterator);
    PyGILState_Release(gstate);

    if (ok) {
        int sample_rate = 24000;
        if (const char* env_sample_rate = std::getenv("TTS_SAMPLE_RATE")) {
            int configured_sample_rate = std::atoi(env_sample_rate);
            if (configured_sample_rate > 0) {
                sample_rate = configured_sample_rate;
            }
        }
        fprintf(stdout, "[TtsEngine] Stream synthesized %zu samples in %d chunks (%.2f sec)\n",
                total_samples, chunk_count, (double)total_samples / (double)sample_rate);
    } else {
        fprintf(stderr, "[TtsEngine] Stream synthesis failed or interrupted after %zu samples\n",
                total_samples);
    }
    return ok && total_samples > 0;
}

// ============================================================
// 异步合成
// ============================================================

void TtsEngine::synthesize_async(const std::string& text,
                                 const std::string& spk_id,
                                 TtsEngineDoneCallback on_done) {
    if (!m_initialized) {
        fprintf(stderr, "[TtsEngine] Not initialized, cannot queue async task!\n");
        if (on_done) on_done(false, text, "TtsEngine not initialized");
        return;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pending_text = text;
        m_pending_spk_id = spk_id;
        m_pending_callback = std::move(on_done);
        m_has_pending = true;
        m_cancel = false;
    }

    m_cv.notify_one();
}

// ============================================================
// 等待完成 / 取消 / 忙碌检查
// ============================================================

void TtsEngine::wait_for_done() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this]() { return !m_has_pending; });
}

void TtsEngine::cancel() {
    m_cancel = true;
    m_busy = false;
    m_has_pending = false;
}

bool TtsEngine::is_busy() const {
    return m_busy;
}

// ============================================================
// 后台工作线程
// ============================================================

void TtsEngine::worker_loop() {
    while (m_running) {
        std::string text;
        std::string spk_id;
        TtsEngineDoneCallback callback;

        // 等待任务
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this]() {
                return m_has_pending || !m_running;
            });

            if (!m_running) break;
            if (m_cancel) {
                m_has_pending = false;
                m_cancel = false;
                continue;
            }

            text = m_pending_text;
            spk_id = m_pending_spk_id;
            callback = std::move(m_pending_callback);
            m_has_pending = false;
        }

        m_busy = true;

        fprintf(stdout, "[TtsEngine] Async synthesizing: \"%s\"\n", text.c_str());

        std::vector<int16_t> pcm;
        bool ok = synthesize_sync(text, pcm, spk_id);

        if (callback) {
            callback(ok, text,
                     ok ? ("OK: " + std::to_string(pcm.size()) + " samples")
                        : "Synthesis failed");
        }

        m_busy = false;
        m_cv.notify_one();  // notify wait_for_done()
    }
}
