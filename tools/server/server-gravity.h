#pragma once

#include "server-common.h"
#include "server-task.h"

// gravity: eval mode (fixed defaults, 1 slot, no cache) and the "gravity" provenance object

// force the eval-mode params; zero the global LoRA scales when --lora-init-without-apply or --eval-mode is set
void gravity_prepare_params(common_params & params);

struct gravity_fill {
    json body;                          // request body with defaults written, only set in eval mode
    json filled     = json::array();    // field names the server filled
    json overridden = json::array();    // field names the server forced
};

// eval mode: write the fixed default into every absent field, never refuse
gravity_fill gravity_fill_request(const common_params & params, const json & data);

// provenance of one request, without "lora" (see gravity_set_lora)
json gravity_provenance(const common_params & params, const task_params & tp, const gravity_fill & gf);

// add the "lora" scales the slot ran with
void gravity_set_lora(json & gravity, const common_params & params, const std::vector<common_adapter_lora_info> & lora);

// static part for GET /props
json gravity_props(const common_params & params);

// add the object to a final completion result (the last chunk when the result is a list of chunks)
json gravity_attach(json res, task_response_type res_type, const json & gravity);
