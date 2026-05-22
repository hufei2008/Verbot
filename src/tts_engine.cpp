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

    // 析构函数中不做任何 Python 操作。Python 资源在 shutdown() 中释放，
    // 必须在 main() 返回前、Python atexit 处理程序执行之前调用。
}

// ============================================================
// shutdown — 停止 TTS Engine（不终结 Python 解释器）
//
// 只做安全退出所需的最小操作：
//   1. 停止 C++ 工作线程（不再执行 Python 调用）
//   2. 清空 Python 指针（标记不可用）
//
// 不调用 Py_FinalizeEx() 或任何 Python C API 清理函数。
// 原因：cosyvoice_bridge、sentencepiece、numpy 等 C 扩展在 Python
// finalization 阶段（Py_FinalizeEx/exit/return）内部可能创建了
// 额外的 C/C++ 线程或注册了线程状态，导致 threading._shutdown()
// 调用 PyThreadState_Get 时 crash。
//
// 嵌入式 Python 的推荐做法：让 OS 在进程退出时回收所有资源，
// 这比手动调用 Py_FinalizeEx() 更安全。
//
// main() 调用本函数后必须使用 _exit() 退出进程，跳过 exit()
// 的 atexit 处理程序和 C++ 析构函数执行链。
// ============================================================

void TtsEngine::shutdown() {
    if (!m_python_initialized && !m_py_bridge_module) {
        return;
    }

    // 停止工作线程
    if (m_worker.joinable()) {
        m_running = false;
        m_cv.notify_one();
        m_worker.join();
    }

    // 不调用任何 Python C API 清理函数（包括 Py_FinalizeEx）。
    // 原因：Python C 扩展（sentencepiece、transformers、numpy 等）
    // 在 Python finalization 阶段可能创建额外的线程状态或访问
    // GIL，导致 threading._shutdown() → PyThreadState_Get crash。
    // multiprocessing.resource_tracker daemon 线程也是问题来源。
    // 此外，Python 3.10 中 setenv("PYTHONMULTIPROCESSING_RESOURCE_TRACKER","0")
    // 无效（该 env var 从 Python 3.12 才引入）。
    //
    // 安全的做法：让 OS 回收所有资源。

    m_py_bridge_module = nullptr;
    m_python_main_thread_state = nullptr;
    m_python_gil_released = false;
    m_python_initialized = false;
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

    // ★ 在 bridge 模块完全加载后，永久禁用 multiprocessing.resource_tracker。
    //
    // cosyvoice_bridge 的导入链（transformers → huggingface_hub → filelock
    // → threading → multiprocessing）可能触发 resource_tracker daemon 线程
    // 的创建。这个 daemon 线程在后台运行 Python 代码并等待 pipe 输入，
    // 在主线程释放 GIL 时它可被唤醒，访问 PyThreadState 导致 crash。
    //
    // setenv("PYTHONMULTIPROCESSING_RESOURCE_TRACKER","0") 在 Python 3.10
    // 中无效（该 env var 从 3.12 才被识别）。_stop() 也不可靠，因为：
    // - 在 _stop() 调用后其他代码又 import multiprocessing 时会重新创建
    // - daemon 线程退出需要时间，可能仍晚于崩溃发生
    //
    // 终极方案（下面实现）：
    // 1. 停止现有 daemon 线程
    // 2. 替换 ensure_running() 为 no-op，使 daemon 永远无法启动
    // 3. 替换 register/unregister 为 no-op
    // 4. 设置 _resource_tracker = None
    // 5. 过滤 warnings
    {
        PyGILState_STATE gstate = PyGILState_Ensure();
        PyRun_SimpleString(
            "import sys\n"
            "if 'multiprocessing.resource_tracker' in sys.modules:\n"
            "    import multiprocessing.resource_tracker as _rt\n"
            "    # 1. 停止现有的 daemon 线程\n"
            "    if _rt._resource_tracker is not None:\n"
            "        try:\n"
            "            _rt._resource_tracker._stop()\n"
            "        except Exception:\n"
            "            pass\n"
            "    # 2. 永久禁用：替换核心函数为 no-op\n"
            "    _rt.ensure_running = lambda: None\n"
            "    _rt.register = lambda *a, **kw: None\n"
            "    _rt.unregister = lambda *a, **kw: None\n"
            "    _rt._resource_tracker = None\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore', message='resource_tracker:.*')\n"
        );
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

    // 在初始化 Python 解释器之前设置环境变量，禁用
    // multiprocessing.resource_tracker daemon 线程。
    // bridge 模块（mlx、transformers、sentencepiece 等）可能间接导入
    // multiprocessing，其 resource_tracker 会创建一个 daemon 线程。
    // 进程退出时该 daemon 线程仍然会运行，访问 GIL 状态导致
    // Fatal Python error: PyThreadState_Get crash。
    // 必须用 C 的 setenv() 在 Py_Initialize() 之前设置，
    // 因为 Python 启动后 os.environ 设置可能已经太晚。
    setenv("PYTHONMULTIPROCESSING_RESOURCE_TRACKER", "0", 1);
    setenv("PYTHONWARNINGS", "ignore:resource_tracker:UserWarning", 0);

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
        ("import sys\n"
         "sys.executable = '" + python_exe + "'\n").c_str());

    // 设置环境变量禁用 multiprocessing.resource_tracker daemon 线程。
    // bridge 模块（mlx、transformers、sentencepiece 等）可能间接导入
    // multiprocessing，其 resource_tracker 会创建一个 daemon 线程。
    // 进程退出时该 daemon 线程仍然会运行，并访问 GIL 状态导致
    // Fatal Python error: PyThreadState_Get crash。
    // 环境变量方式比事后 stop() 更可靠，因为它阻止了这个线程的创建。
    PyRun_SimpleString(
        "import os\n"
        "os.environ['PYTHONMULTIPROCESSING_RESOURCE_TRACKER'] = '0'\n"
    );

    // 验证关键模块可导入
    int ret = PyRun_SimpleString(
        "try:\n"
        "    import numpy\n"
        "    print(f'[TtsEngine] numpy {numpy.__version__} loaded')\n"
        "except Exception as e:\n"
        "    print(f'[TtsEngine] WARNING: numpy import failed: {e}')\n"
    );
    (void)ret;

    // ★ 终结 multiprocessing.resource_tracker 守护线程！
    //
    // 这是最关键的一步。Python 3.10 的 multiprocessing 模块
    // 默认创建一个守护线程（resource_tracker）来跟踪命名信号量。
    // 这个 daemon 线程在后台运行 Python 代码，当主线程释放 GIL
    // 时它可能被唤醒，导致 GIL 状态不一致 crash。
    //
    // setenv("PYTHONMULTIPROCESSING_RESOURCE_TRACKER","0") 在
    // Python 3.10 中不起作用（该 env var 从 3.12 才引入），
    // 所以只能运行时通过 Python 代码来 stop 它。
    //
    // 正确做法：
    // 1. 调用 rt._resource_tracker._stop() 发送停止信号到管道
    // 2. daemon 线程收到后退出循环
    // 3. 设置 rt._resource_tracker = None 防止后续重新启动
    // 4. 抑制 resource_tracker 的警告信息
    PyRun_SimpleString(
        "import sys\n"
        "if 'multiprocessing.resource_tracker' in sys.modules:\n"
        "    import multiprocessing.resource_tracker as rt\n"
        "    if rt._resource_tracker is not None:\n"
        "        try:\n"
        "            rt._resource_tracker._stop()\n"
        "        except Exception:\n"
        "            pass\n"
        "        rt._resource_tracker = None\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore', message='resource_tracker:.*')\n"
    );

    m_python_initialized = true;
    // 释放主线程持有的 GIL。TTS 合成发生在独立线程里，如果这里不
    // 释放 GIL，播放线程会卡在 PyGILState_Ensure()，主线程又在
    // wait_for_tts_for() 等它结束，形成死锁。
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

    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this]() { return !m_has_pending && !m_busy; });
        m_pending_kind = 1;
        m_pending_text = text;
        m_pending_spk_id = spk_id;
        m_result_pcm.clear();
        m_request_done = false;
        m_request_ok = false;
        m_has_pending = true;
        m_cancel = false;
    }
    m_cv.notify_one();

    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this]() { return m_request_done; });
    out_pcm = std::move(m_result_pcm);
    return m_request_ok;
}

bool TtsEngine::synthesize_sync_python(const std::string& text,
                                       std::vector<int16_t>& out_pcm,
                                       const std::string& spk_id) {
    out_pcm.clear();

    PyObject* bridge = static_cast<PyObject*>(m_py_bridge_module);

    // 调用 bridge.synthesize(text, spk_id, "sft")
    PyObject* result = PyObject_CallMethod(bridge, "synthesize", "(sss)",
                                           text.c_str(),
                                           spk_id.c_str(),
                                           "sft");
    if (!result) {
        PyErr_Print();
        fprintf(stderr, "[TtsEngine] bridge.synthesize() failed!\n");
        return false;
    }

    // 通过 numpy.ndarray.tobytes() 获取数据（通用方式，不依赖 numpy 头文件细节）
    PyObject* buffer = PyObject_CallMethod(result, "tobytes", nullptr);
    if (!buffer) {
        PyErr_Print();
        Py_DECREF(result);
        fprintf(stderr, "[TtsEngine] result.tobytes() call failed!\n");
        return false;
    }

    // 获取字节数据
    const char* buf_data = nullptr;
    Py_ssize_t buf_len = 0;
    if (PyBytes_AsStringAndSize(buffer, const_cast<char**>(&buf_data), &buf_len) == -1) {
        Py_DECREF(buffer);
        Py_DECREF(result);
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

    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this]() { return !m_has_pending && !m_busy; });
        m_pending_kind = 2;
        m_pending_text = text;
        m_pending_spk_id = spk_id;
        m_pending_stream_callback = std::move(on_chunk);
        m_request_done = false;
        m_request_ok = false;
        m_has_pending = true;
        m_cancel = false;
    }
    m_cv.notify_one();

    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this]() { return m_request_done; });
    return m_request_ok;
}

bool TtsEngine::synthesize_stream_python(const std::string& text,
                                         const std::string& spk_id,
                                         TtsStreamChunkCallback on_chunk) {
    PyObject* bridge = static_cast<PyObject*>(m_py_bridge_module);
    PyObject* generator = PyObject_CallMethod(bridge, "synthesize_stream", "(sss)",
                                              text.c_str(),
                                              spk_id.c_str(),
                                              "sft");
    if (!generator) {
        PyErr_Print();
        fprintf(stderr, "[TtsEngine] bridge.synthesize_stream() failed!\n");
        return false;
    }

    PyObject* iterator = PyObject_GetIter(generator);
    Py_DECREF(generator);
    if (!iterator) {
        PyErr_Print();
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

        if (!pcm.empty()) {
            if (!on_chunk(pcm)) {
                ok = false;
                break;
            }
        }
    }

    Py_DECREF(iterator);

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
        if (m_has_pending || m_busy) {
            if (on_done) on_done(false, text, "TtsEngine busy");
            return;
        }
        m_pending_kind = 3;
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
    m_cv.wait(lock, [this]() { return !m_has_pending && !m_busy; });
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
    PyGILState_STATE worker_gstate = PyGILState_Ensure();
    (void)worker_gstate;

    while (m_running) {
        std::string text;
        std::string spk_id;
        TtsEngineDoneCallback callback;
        TtsStreamChunkCallback stream_callback;
        int kind = 0;

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
                m_request_done = true;
                m_request_ok = false;
                m_cv.notify_all();
                continue;
            }

            kind = m_pending_kind;
            text = m_pending_text;
            spk_id = m_pending_spk_id;
            callback = std::move(m_pending_callback);
            stream_callback = std::move(m_pending_stream_callback);
            m_has_pending = false;
            m_busy = true;
        }

        bool ok = false;
        std::vector<int16_t> pcm;

        if (kind == 2) {
            fprintf(stdout, "[TtsEngine] Worker streaming: \"%s\"\n", text.c_str());
            ok = synthesize_stream_python(text, spk_id, stream_callback);
        } else {
            if (kind == 3) {
                fprintf(stdout, "[TtsEngine] Async synthesizing: \"%s\"\n", text.c_str());
            }
            ok = synthesize_sync_python(text, pcm, spk_id);
        }

        if (callback) {
            callback(ok, text,
                     ok ? ("OK: " + std::to_string(pcm.size()) + " samples")
                        : "Synthesis failed");
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (kind == 1) {
                m_result_pcm = std::move(pcm);
            }
            m_pending_kind = 0;
            m_request_ok = ok;
            m_request_done = true;
            m_busy = false;
        }
        m_cv.notify_all();
    }
}
