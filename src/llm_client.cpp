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

bool LlmClient::load_model(const std::string& model_path, int n_ctx, int n_threads, bool use_gpu) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_model) {
        unload_model_locked();
    }

    m_model_path = model_path;
    m_n_ctx      = n_ctx;
    m_n_threads  = n_threads;
    m_use_gpu    = use_gpu;

    // 1. 后端初始化
    llama_backend_init();

    // 2. 模型参数 — n_gpu_layers 在这里设置
    auto model_params = llama_model_default_params();
    model_params.use_mmap  = true;
    model_params.use_mlock = false;
    if (use_gpu) {
        model_params.n_gpu_layers = -1;  // 全部层卸载到 GPU
    } else {
        model_params.n_gpu_layers = 0;
    }

    // 3. 加载模型
    m_model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!m_model) {
        fprintf(stderr, "[LLM] Failed to load model: %s\n", model_path.c_str());
        return false;
    }

    // 4. 获取 vocab — const 指针，需要 const_cast
    const struct llama_vocab * vocab_const = llama_model_get_vocab(m_model);
    m_vocab = const_cast<struct llama_vocab *>(vocab_const);
    if (!m_vocab) {
        fprintf(stderr, "[LLM] Failed to get vocab\n");
        llama_model_free(m_model);
        m_model = nullptr;
        return false;
    }

    // 5. 上下文参数 — 注意字段名已更新
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = n_ctx;
    ctx_params.n_batch         = 512;
    ctx_params.n_ubatch        = 512;
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    ctx_params.offload_kqv     = use_gpu;

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

LlmResult LlmClient::run_inference(const LlmInferenceParams& params) {
    LlmResult result;
    result.success = false;

    auto t0 = std::chrono::high_resolution_clock::now();

    llama_memory_clear(llama_get_memory(m_ctx), true);

    // 1. Tokenize 输入 prompt
    const int n_prompt_tokens = 1024;
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    int n_tokens = llama_tokenize(
        m_vocab,
        params.prompt.c_str(),
        (int)params.prompt.size(),
        prompt_tokens.data(),
        n_prompt_tokens,
        true,   // add_special = true (add BOS)
        true    // parse_special = true
    );

    if (n_tokens < 0) {
        // 缓冲区不够大，重试
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

    prompt_tokens.resize(n_tokens);

    // 2. 检查上下文长度
    int n_ctx_avail = llama_n_ctx(m_ctx);
    if (n_tokens + params.max_tokens > n_ctx_avail) {
        fprintf(stderr, "[LLM] Warning: prompt (%d) + max_tokens (%d) > ctx (%d), truncating\n",
                n_tokens, params.max_tokens, n_ctx_avail);
        const int keep_tokens = std::max(1, n_ctx_avail - params.max_tokens);
        if ((int)prompt_tokens.size() > keep_tokens) {
            prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.end() - keep_tokens);
            n_tokens = (int)prompt_tokens.size();
        }
    }

    // 3. 创建 Sampler Chain
    if (m_sampler) {
        llama_sampler_free(m_sampler);
        m_sampler = nullptr;
    }

    auto sparams = llama_sampler_chain_default_params();
    m_sampler = llama_sampler_chain_init(sparams);

    if (params.temperature < 0.01f) {
        llama_sampler_chain_add(m_sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(m_sampler, llama_sampler_init_top_k((int)params.top_k));
        llama_sampler_chain_add(m_sampler, llama_sampler_init_top_p(params.top_p, 1));
        llama_sampler_chain_add(m_sampler, llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(m_sampler, llama_sampler_init_penalties(64, params.repeat_penalty, 0.0f, 0.0f));
        llama_sampler_chain_add(m_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    // 4. 解码 prompt (prefill)
    const int n_batch = std::max(1, (int)llama_n_batch(m_ctx));
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

    // 5. 自回归生成
    std::string output;
    int n_generated = 0;
    for (int i = 0; i < params.max_tokens; ++i) {
        // 采样
        llama_token new_token_id = llama_sampler_sample(m_sampler, m_ctx, -1);

        if (llama_vocab_is_eog(m_vocab, new_token_id)) {
            break; // 遇到 EOS 停止
        }

        // 解码 token → text
        char piece[32] = {0};
        int n_bytes = llama_token_to_piece(
            m_vocab,
            new_token_id,
            piece,
            sizeof(piece),
            0,    // lstrip
            true  // special
        );

        if (n_bytes > 0) {
            output.append(piece, n_bytes);
            n_generated++;
        }

        if (output.find("<end_of_turn>") != std::string::npos ||
            output.find("```") != std::string::npos) {
            break;
        }

        // 准备下一个 batch (单个 token)
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
