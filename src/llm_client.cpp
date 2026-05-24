#include "llm_client.h"
#include <cstdio>
#include <cstring>
#include <sstream>
#include <cstdlib>
#include <algorithm>

LlmClient::LlmClient() {
    // 保留默认日志，便于调试
}

LlmClient::~LlmClient() {
    unload_model();
}

// ============================================================
// 模型加载流程：
// 1. 初始化 llama 后端
// 2. 配置模型参数（内存映射、GPU 层数）
// 3. 加载模型文件
// 4. 获取词表（vocab）
// 5. 创建推理上下文
// ============================================================
bool LlmClient::load_model(const std::string& model_path, int n_ctx, int n_threads, bool use_gpu) {
    std::lock_guard<std::mutex> lock(m_mutex);

    // 如果已有加载的模型，先卸载
    if (m_model) {
        unload_model_locked();
    }

    m_model_path = model_path;
    m_n_ctx      = n_ctx;
    m_n_threads  = n_threads;
    m_use_gpu    = use_gpu;

    // 1. 后端初始化（全局，可多次调用）
    llama_backend_init();

    // 2. 模型参数 — n_gpu_layers 在这里设置
    auto model_params = llama_model_default_params();
    model_params.use_mmap  = true;   // 使用内存映射加速加载
    model_params.use_mlock = false;  // 不锁定内存页
    if (use_gpu) {
        model_params.n_gpu_layers = -1;  // 全部层卸载到 GPU
    } else {
        model_params.n_gpu_layers = 0;   // 仅使用 CPU
    }

    // 3. 加载模型文件
    m_model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!m_model) {
        fprintf(stderr, "[LLM] Failed to load model: %s\n", model_path.c_str());
        return false;
    }

    // 4. 获取 vocab — const 指针，需要 const_cast 转为可变指针
    const struct llama_vocab * vocab_const = llama_model_get_vocab(m_model);
    m_vocab = const_cast<struct llama_vocab *>(vocab_const);
    if (!m_vocab) {
        fprintf(stderr, "[LLM] Failed to get vocab\n");
        llama_model_free(m_model);
        m_model = nullptr;
        return false;
    }

    // 5. 上下文参数 — 配置推理上下文大小、批处理、线程等
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = n_ctx;                        // 上下文窗口大小
    ctx_params.n_batch         = 512;                          // 批处理大小
    ctx_params.n_ubatch        = 512;                          // 微批处理大小
    ctx_params.n_threads       = n_threads;                    // 推理线程数
    ctx_params.n_threads_batch = n_threads;                    // 批处理线程数
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;   // Flash Attention 自动模式
    ctx_params.offload_kqv     = use_gpu;                      // 将 KQV 矩阵卸载到 GPU

    m_ctx = llama_init_from_model(m_model, ctx_params);
    if (!m_ctx) {
        fprintf(stderr, "[LLM] Failed to create context\n");
        llama_model_free(m_model);
        m_model = nullptr;
        m_vocab = nullptr;
        return false;
    }

    fprintf(stdout, "[LLM] Model loaded: %s\n", model_info().c_str());
    return true;
}

void LlmClient::unload_model() {
    std::lock_guard<std::mutex> lock(m_mutex);
    unload_model_locked();
}

void LlmClient::unload_model_locked() {
    if (m_sampler) {
        llama_sampler_free(m_sampler);
        m_sampler = nullptr;
    }
    if (m_ctx) {
        llama_free(m_ctx);
        m_ctx = nullptr;
    }
    if (m_model) {
        llama_model_free(m_model);
        m_model = nullptr;
    }
    m_vocab = nullptr;
}

bool LlmClient::is_loaded() const {
    return m_model != nullptr && m_ctx != nullptr;
}

std::string LlmClient::model_info() const {
    if (!m_model) return "(no model)";
    char desc[256] = {0};
    llama_model_desc(m_model, desc, sizeof(desc));
    uint64_t n_params = llama_model_n_params(m_model);
    uint64_t size     = llama_model_size(m_model);
    char buf[512];
    snprintf(buf, sizeof(buf), "%s | %.1fB params | %.1f GB | ctx=%d",
             desc, n_params / 1e9f, size / (1024.0*1024.0*1024.0), m_n_ctx);
    return std::string(buf);
}

std::string LlmClient::model_name() const {
    if (!m_model) return "(no model)";
    char desc[256] = {0};
    llama_model_desc(m_model, desc, sizeof(desc));
    return std::string(desc);
}

std::string LlmClient::format_chat_prompt(const std::vector<std::pair<std::string, std::string>>& messages) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_model || messages.empty()) return "";

    const char* tmpl = llama_model_chat_template(m_model, nullptr);
    if (tmpl == nullptr) {
        std::ostringstream oss;
        std::string pending_system;
        oss << "<bos>";
        for (const auto& msg : messages) {
            if (msg.first == "system") {
                if (!pending_system.empty()) pending_system += "\n\n";
                pending_system += msg.second;
                continue;
            }

            std::string role = msg.first == "assistant" ? "model" : msg.first;
            oss << "<|turn>" << role << "\n";
            if (role == "user" && !pending_system.empty()) {
                oss << pending_system << "\n\n";
                pending_system.clear();
            }
            oss << msg.second << "\n";
        }
        oss << "<|turn>model\n<|channel>thought\n";
        return oss.str();
    }

    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    for (const auto& msg : messages) {
        chat.push_back({msg.first.c_str(), msg.second.c_str()});
    }

    int32_t len = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, nullptr, 0);
    if (len < 0) {
        fprintf(stderr, "[LLM] Failed to apply model chat template\n");
        return "";
    }

    std::vector<char> buffer((size_t)len + 1, '\0');
    int32_t written = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true,
                                                buffer.data(), (int32_t)buffer.size());
    if (written < 0) {
        fprintf(stderr, "[LLM] Failed to format chat prompt\n");
        return "";
    }

    return std::string(buffer.data(), (size_t)written);
}

LlmResult LlmClient::infer_sync(const LlmInferenceParams& params) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_model || !m_ctx) {
        LlmResult r;
        r.success   = false;
        r.error_msg = "Model not loaded";
        return r;
    }
    return run_inference(params);
}

// ============================================================
// 核心推理流程：
// 1. Tokenize - 将文本 prompt 转换为 token ID 序列
// 2. 上下文检查 - 确保 prompt + max_tokens 不超出上下文窗口
// 3. 采样器链 - 配置 top-k/top-p/temperature/repeat_penalty 采样策略
// 4. Prefill - 预填充阶段，一次性处理所有 prompt tokens
// 5. 自回归生成 - 逐个 token 生成，直到 EOS 或达到 max_tokens
// ============================================================
LlmResult LlmClient::run_inference(const LlmInferenceParams& params) {
    LlmResult result;
    result.success = false;

    auto t0 = std::chrono::high_resolution_clock::now();  // 开始计时

    // 清空之前的推理内存，释放 KV cache
    llama_memory_clear(llama_get_memory(m_ctx), true);

    // ============================================================
    // 1. Tokenize 输入 prompt（文本 → token ID 序列）
    // ============================================================
    const int n_prompt_tokens = 1024;                       // 初始缓冲区大小
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    int n_tokens = llama_tokenize(
        m_vocab,
        params.prompt.c_str(),
        (int)params.prompt.size(),
        prompt_tokens.data(),
        n_prompt_tokens,
        true,   // add_special = true (添加 BOS 标记)
        true    // parse_special = true (解析特殊 token)
    );

    if (n_tokens < 0) {
        // 缓冲区不够大，根据返回值重新分配并重试
        prompt_tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(
            m_vocab,
            params.prompt.c_str(),
            (int)params.prompt.size(),
            prompt_tokens.data(),
            (int)prompt_tokens.size(),
            true, true
        );
    }

    if (n_tokens <= 0) {
        result.error_msg = "Failed to tokenize prompt";
        return result;
    }

    prompt_tokens.resize(n_tokens);  // 调整为实际 token 数量

    // ============================================================
    // 2. 检查上下文长度，必要时截断 prompt
    // ============================================================
    int n_ctx_avail = llama_n_ctx(m_ctx);  // 可用的上下文窗口大小
    if (n_tokens + params.max_tokens > n_ctx_avail) {
        fprintf(stderr, "[LLM] Warning: prompt (%d) + max_tokens (%d) > ctx (%d), truncating\n",
                n_tokens, params.max_tokens, n_ctx_avail);
        // 保留最近的 tokens，为生成预留空间
        const int keep_tokens = std::max(1, n_ctx_avail - params.max_tokens);
        if ((int)prompt_tokens.size() > keep_tokens) {
            prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.end() - keep_tokens);
            n_tokens = (int)prompt_tokens.size();
        }
    }

    // ============================================================
    // 3. 创建 Sampler Chain（采样策略链）
    // 根据 temperature 决定使用贪婪解码还是随机采样
    // ============================================================
    if (m_sampler) {
        llama_sampler_free(m_sampler);  // 清理旧采样器
        m_sampler = nullptr;
    }

    auto sparams = llama_sampler_chain_default_params();
    m_sampler = llama_sampler_chain_init(sparams);

    if (params.temperature < 0.01f) {
        // 贪婪解码：每次选择概率最高的 token
        llama_sampler_chain_add(m_sampler, llama_sampler_init_greedy());
    } else {
        // 随机采样链：top-k → top-p → temperature → repeat_penalty → 分布采样
        llama_sampler_chain_add(m_sampler, llama_sampler_init_top_k((int)params.top_k));
        llama_sampler_chain_add(m_sampler, llama_sampler_init_top_p(params.top_p, 1));
        llama_sampler_chain_add(m_sampler, llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(m_sampler, llama_sampler_init_penalties(64, params.repeat_penalty, 0.0f, 0.0f));
        llama_sampler_chain_add(m_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    // ============================================================
    // 4. 解码 prompt (Prefill 预填充阶段)
    // 将所有 prompt tokens 一次性送入模型计算 KV cache
    // ============================================================
    const int n_batch = std::max(1, (int)llama_n_batch(m_ctx));  // 单批最大 token 数
    llama_batch batch{};
    for (int pos = 0; pos < n_tokens; pos += n_batch) {
        const int n_eval = std::min(n_batch, n_tokens - pos);
        batch = llama_batch_get_one(prompt_tokens.data() + pos, n_eval);
        if (llama_decode(m_ctx, batch) != 0) {
            result.error_msg = "Failed to decode prompt";
            llama_sampler_free(m_sampler);
            m_sampler = nullptr;
            return result;
        }
    }

    // ============================================================
    // 5. 自回归生成（逐个 token 循环生成）
    // 每次采样一个 token，将其附加到上下文，继续生成下一个
    // ============================================================
    std::string output;
    int n_generated = 0;
    for (int i = 0; i < params.max_tokens; ++i) {
        // 采样下一个 token
        llama_token new_token_id = llama_sampler_sample(m_sampler, m_ctx, -1);

        // 遇到 EOS（End of Sequence）则停止
        if (llama_vocab_is_eog(m_vocab, new_token_id)) {
            break;
        }

        // token ID → 文本片段
        char piece[32] = {0};
        int n_bytes = llama_token_to_piece(
            m_vocab,
            new_token_id,
            piece,
            sizeof(piece),
            0,    // lstrip = 0，不从左侧去除空格
            true  // special = true，包含特殊 token
        );

        if (n_bytes > 0) {
            output.append(piece, n_bytes);
            n_generated++;
        }

        // 遇到 end_of_turn 标记或 markdown 代码块标记则提前停止
        if (output.find("<end_of_turn>") != std::string::npos ||
            output.find("```") != std::string::npos) {
            break;
        }

        // 将新生成的 token 送入模型进行下一轮计算
        batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(m_ctx, batch) != 0) {
            fprintf(stderr, "[LLM] Decode failed at token %d\n", i);
            break;
        }

        if (n_bytes == 0) {
            fprintf(stderr, "[LLM] Empty token piece at step %d: id=%d\n", i, new_token_id);
        }
    }

    // 清理 sampler
    llama_sampler_free(m_sampler);
    m_sampler = nullptr;

    // 计算耗时并填充结果
    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsed_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.success     = !output.empty();
    result.text        = output;
    result.token_count = n_generated;
    if (output.empty()) {
        result.error_msg = "Model generated no text";
    }

    return result;
}
