#pragma once

#include "server-common.h"
#include "server-task.h"

// gravity: eval mode (fixed defaults, 1 slot, no cache) and the "gravity" provenance object

// gravity release version (CHANGELOG.md); "build" stays upstream llama.cpp build info
#define GRAVITY_VERSION "0.1.0"

// force the eval-mode params; zero the global LoRA scales when --lora-init-without-apply or --eval-mode is set
void gravity_prepare_params(common_params & params);

struct gravity_fill {
    json body;                          // request body with defaults written, only set in eval mode
    json filled     = json::array();    // field names the server filled
    json overridden = json::array();    // field names the server forced
};

// eval mode: write the fixed default into every absent field, never refuse
// only for /completion, /completions, /v1/completions and /v1/chat/completions
gravity_fill gravity_fill_request(const common_params & params, const json & data, server_task_type type, task_response_type res_type);

// provenance of one request, without "lora" (see gravity_set_lora); null outside eval mode, so responses stay upstream's
json gravity_provenance(const common_params & params, const json & devices, const task_params & tp, const gravity_fill & gf);

// add the "lora" scales the slot ran with
void gravity_set_lora(json & gravity, const common_params & params, const std::vector<common_adapter_lora_info> & lora);

// where the loaded model runs: {"devices_used", "device_info", "layers_on_device", "output_device"}, from llama_model_layer_device()
json gravity_devices_used(const llama_model * model);

// static part for GET /props; devices = gravity_devices_used() of the loaded model
json gravity_props(const common_params & params, const json & devices);

// eval mode: answer 400 on every route that could run the model (or load KV) without filled defaults and the echo:
// POST /lora-adapters, POST /slots/:id, /infill, /v1/responses, /v1/messages, embeddings, rerank, /v1/systemone, audio
struct server_routes;
void gravity_restrict_routes(const common_params & params, server_routes & routes);

// add the object to a final completion result (the last chunk when the result is a list of chunks)
json gravity_attach(json res, task_response_type res_type, const json & gravity);
