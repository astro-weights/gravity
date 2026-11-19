#include "server-gravity.h"

#include "build-info.h"
#include "sampling.h"

// eval-mode defaults, listed in tools/server/README.md and --help: change = new gravity version
static const int    GRAVITY_SEED        = 0;
static const double GRAVITY_TEMPERATURE = 0.0;
static const int    GRAVITY_TOP_K       = 1;
static const char * GRAVITY_SAMPLERS[]  = { "top_k", "temperature" };

void gravity_prepare_params(common_params & params) {
    if (params.eval_mode) {
        if (params.n_parallel > 1 || params.n_cache_reuse > 0) {
            SRV_WRN("--eval-mode overrides n_parallel = %d and n_cache_reuse = %d to 1 and 0\n", params.n_parallel, params.n_cache_reuse);
        }
        params.n_parallel    = 1;
        params.n_cache_reuse = 0;
        params.cache_prompt  = false;
        params.cont_batching = false;
        params.lora_init_without_apply = true;
    }
    if (params.lora_init_without_apply) {
        for (auto & lora : params.lora_adapters) {
            lora.scale = 0.0f;
        }
    }
}

static bool is_absent(const json & body, const char * key) {
    return !body.contains(key) || body.at(key).is_null();
}

gravity_fill gravity_fill_request(const common_params & params, const json & data) {
    gravity_fill gf;
    if (!params.eval_mode || !data.is_object()) {
        return gf;
    }
    gf.body = data;
    auto fill = [&](const char * key, const json & value) {
        if (is_absent(gf.body, key)) {
            gf.body[key] = value;
            gf.filled.push_back(key);
        }
    };
    fill("seed",        GRAVITY_SEED);
    fill("temperature", GRAVITY_TEMPERATURE);
    fill("samplers",    json(std::vector<std::string>(std::begin(GRAVITY_SAMPLERS), std::end(GRAVITY_SAMPLERS))));
    fill("top_k",       GRAVITY_TOP_K);

    // an empty list means "global config" for the server, treat it as absent
    const bool lora_empty = gf.body.contains("lora") && gf.body.at("lora").is_array() && gf.body.at("lora").empty();
    if (!params.lora_adapters.empty() && (is_absent(gf.body, "lora") || lora_empty)) {
        json lora = json::array();
        for (size_t i = 0; i < params.lora_adapters.size(); i++) {
            lora.push_back({ {"id", i}, {"scale", 0.0} });
        }
        gf.body["lora"] = lora;
        gf.filled.push_back("lora");
    }

    if (gf.body.value("cache_prompt", false)) {
        gf.body["cache_prompt"] = false;
        gf.overridden.push_back("cache_prompt");
    }
    return gf;
}

static json pin_or_null(const std::string & pin) {
    return pin.empty() ? json(nullptr) : json(pin);
}

static json backend_names(const common_params & params) {
    json names = json::array();
    if (params.devices.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            names.push_back(ggml_backend_dev_name(ggml_backend_dev_get(i)));
        }
    }
    for (auto * dev : params.devices) {
        if (dev) {
            names.push_back(ggml_backend_dev_name(dev));
        }
    }
    return names;
}

static json lora_json(const common_params & params, size_t i, float scale) {
    return {
        {"id",     i},
        {"scale",  scale},
        {"sha256", pin_or_null(i < params.lora_sha256.size() ? params.lora_sha256[i] : "")},
    };
}

json gravity_props(const common_params & params) {
    json lora = json::array();
    for (size_t i = 0; i < params.lora_adapters.size(); i++) {
        lora.push_back(lora_json(params, i, params.lora_adapters[i].scale));
    }
    return {
        {"eval_mode",    params.eval_mode},
        {"build",        std::string(llama_build_info())},
        {"backend",      backend_names(params)},
        {"model_sha256", pin_or_null(params.model_sha256)},
        {"lora",         lora},
    };
}

json gravity_provenance(const common_params & params, const task_params & tp, const gravity_fill & gf) {
    json g = gravity_props(params);
    g.erase("lora");
    json samplers = json::array();
    for (auto s : tp.sampling.samplers) {
        samplers.push_back(common_sampler_type_to_str(s));
    }
    g["seed"]        = tp.sampling.seed;
    g["temperature"] = tp.sampling.temp;
    g["samplers"]    = samplers;
    g["filled"]      = gf.filled;
    g["overridden"]  = gf.overridden;
    return g;
}

void gravity_set_lora(json & gravity, const common_params & params, const std::vector<common_adapter_lora_info> & lora) {
    if (gravity.is_null()) {
        return;
    }
    json arr = json::array();
    for (size_t i = 0; i < lora.size(); i++) {
        arr.push_back(lora_json(params, i, lora[i].scale));
    }
    gravity["lora"] = arr;
}

json gravity_attach(json res, task_response_type res_type, const json & gravity) {
    if (gravity.is_null() || (res_type != TASK_RESPONSE_TYPE_NONE && res_type != TASK_RESPONSE_TYPE_OAI_CMPL && res_type != TASK_RESPONSE_TYPE_OAI_CHAT)) {
        return res;
    }
    json & target = res.is_array() ? res.back() : res;
    target["gravity"] = gravity;
    return res;
}
