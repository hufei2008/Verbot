#ifndef TTS_ENGINE_H
#define TTS_ENGINE_H

#include <string>
#include <vector>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>

// ============================================================
// TtsEngine — Embedded Python CosyVoice TTS Engine
//
// 将 CosyVoice 模型直接嵌入 C++ 进程内部：
//   - 使用 Embedded Python 解释器加载 cosyvoice_bridge.py
//   - 进程启动时加载模型一次，常驻内存
//   - synthesize() 直接调用 Python 侧函数返回 numpy 数组
//   - 零网络开销，零进程间通信
//
// 依赖：
//   - conda env: cosyvoice (Python 3.10 + torch + cosyvoice)
//   - CMake 链接: /path/to/libpython3.10.dylib
//
// 线程安全：
//   - Python GIL 由 PyGILState_Ensure()/Release() 管理
//   - 后台工作线程处理合成请求
// ============================================================

// TTS 完成回调
using TtsEngineDoneCallback = std::function<void(bool success, const std::string& text, const std::string& msg)>;
using TtsStreamChunkCallback = std::function<bool(const std::vector<int16_t>& pcm_chunk)>;

class TtsEngine {
public:
    TtsEngine();
    ~TtsEngine();

    // ──────────────────────────────────────────────────────
    // 初始化
    // ──────────────────────────────────────────────────────

    /// 初始化嵌入式 Python 解释器并加载 CosyVoice 模型。
    /// @param cosyvoice_model_dir CosyVoice 模型路径（本地路径）
    /// @param python_home         conda Python 环境路径（如 /opt/.../envs/cosyvoice）
    /// @param bridge_script_dir   cosyvoice_bridge.py 所在目录
    /// @return true 成功
    bool init(const std::string& cosyvoice_model_dir,
              const std::string& python_home,
              const std::string& bridge_script_dir);

    /// 是否已初始化
    bool is_initialized() const { return m_initialized; }

    // ──────────────────────────────────────────────────────
    // 同步合成
    // ──────────────────────────────────────────────────────

    /// 同步方式：将文本合成为 PCM 音频数据。
    /// @param text   要合成的文本
    /// @param out_pcm 输出 PCM 数据 (22050Hz, int16, mono)
    /// @param spk_id 说话人 ID
    /// @return true 成功
    bool synthesize_sync(const std::string& text,
                         std::vector<int16_t>& out_pcm,
                         const std::string& spk_id = "default");

    /// 流式方式：每生成一段 PCM 就回调一次。回调返回 false 可中断后续处理。
    bool synthesize_stream(const std::string& text,
                           const std::string& spk_id,
                           TtsStreamChunkCallback on_chunk);

    // ──────────────────────────────────────────────────────
    // 异步合成
    // ──────────────────────────────────────────────────────

    /// 异步方式：合成完成后通过回调返回 PCM 数据。
    /// 调用后立即返回，合成在后台线程中进行。
    void synthesize_async(const std::string& text,
                          const std::string& spk_id = "default",
                          TtsEngineDoneCallback on_done = nullptr);

    /// 等待异步请求完成（阻塞）
    void wait_for_done();

    /// 取消当前请求
    void cancel();

    /// 是否正在处理
    bool is_busy() const;

private:
    // ──────────────────────────────────────────────────────
    // 内部方法
    // ──────────────────────────────────────────────────────

    void worker_loop();
    bool ensure_python_initialized();
    bool load_bridge_module();

    // ──────────────────────────────────────────────────────
    // 状态
    // ──────────────────────────────────────────────────────

    std::string m_model_dir;
    std::string m_python_home;
    std::string m_bridge_script_dir;

    std::atomic<bool> m_initialized{false};
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_busy{false};
    std::atomic<bool> m_cancel{false};

    std::thread m_worker;
    std::mutex m_mutex;
    std::condition_variable m_cv;

    // 待处理任务
    std::string m_pending_text;
    std::string m_pending_spk_id;
    TtsEngineDoneCallback m_pending_callback;
    bool m_has_pending{false};

    // Python 解释器状态
    bool m_python_initialized{false};
    bool m_python_gil_released{false};
    void* m_py_bridge_module{nullptr};  // PyObject* (cosyvoice_bridge module)
};

#endif // TTS_ENGINE_H
