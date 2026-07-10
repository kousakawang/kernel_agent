            # Baseline Run Report

            - service_cmd: `CUDA_VISIBLE_DEVICES=7 SGLANG_VLM_CACHE_SIZE_MB=0 python3 -m sglang.launch_server --model-path /data01/models/Qwen3.5-9B/ --host 127.0.0.1 --port 8080 --mem-fraction-static 0.7 --cuda-graph-max-bs 128 --tensor-parallel-size 1 --mm-attention-backend fa3 --cuda-graph-bs 128 120 112 104 96 88 80 72 64 56 48 40 32 24 16 8 4 2 1  --disable-radix-cache`
            - workload_cmd: `python3 -m sglang.bench_serving --backend sglang-oai-chat --dataset-name image --num-prompts 64 --apply-chat-template --random-output-len 32 --random-input-len 16 --image-resolution 480x720 --image-format jpeg --image-count 1 --image-content random --random-range-ratio 1 --host=127.0.0.1 --port=8080`
            - health_url: `http://127.0.0.1:8080/health`
            - workload_returncode: `0`
            - workload_elapsed_sec: `17.412`

            ## Workload Stdout Tail

            ```text
            benchmark_args=Namespace(backend='sglang-oai-chat', base_url=None, host='127.0.0.1', port=8080, ready_check_timeout_sec=60, dataset_name='image', dataset_path='', speed_bench_category=None, speed_bench_output_len=512, model=None, served_model_name=None, tokenizer=None, num_prompts=64, sharegpt_output_len=None, sharegpt_context_len=None, random_input_len=16, random_output_len=32, random_range_ratio=1.0, image_count=1, image_resolution='480x720', random_image_count=False, image_format='jpeg', image_content='random', request_rate=inf, use_trace_timestamps=False, max_concurrency=None, output_file=None, output_details=False, print_requests=False, disable_tqdm=False, disable_stream=False, return_logprob=False, top_logprobs_num=0, token_ids_logprob=None, logprob_start_len=-1, return_routed_experts=False, seed=1, disable_ignore_eos=False, temperature=0.0, top_p=1.0, extra_request_body=None, apply_chat_template=True, profile=False, plot_throughput=False, profile_activities=['CPU', 'GPU'], profile_start_step=None, profile_steps=None, profile_num_steps=None, profile_by_stage=False, profile_stages=None, profile_output_dir=None, profile_prefix=None, lora_name=None, lora_request_distribution='uniform', lora_zipf_alpha=1.5, prompt_suffix='', pd_separated=False, profile_prefill_url=None, profile_decode_url=None, flush_cache=False, warmup_requests=1, tokenize_prompt=False, gsp_num_groups=64, gsp_prompts_per_group=16, gsp_system_prompt_len=2048, gsp_question_len=128, gsp_output_len=256, gsp_range_ratio=1.0, gsp_fast_prepare=False, gsp_send_routing_key=False, gsp_num_turns=1, gsp_ordered=False, gsp_group_distribution='uniform', gsp_zipf_alpha=None, mooncake_slowdown_factor=1.0, mooncake_num_rounds=1, mooncake_workload='conversation', fake_prefill=False, tag=None, header=None)
Waiting up to 60s for http://127.0.0.1:8080/v1/models to become ready...
Server ready in 0.0s.
Namespace(backend='sglang-oai-chat', base_url=None, host='127.0.0.1', port=8080, ready_check_timeout_sec=60, dataset_name='image', dataset_path='', speed_bench_category=None, speed_bench_output_len=512, model='/data01/models/Qwen3.5-9B/', served_model_name=None, tokenizer=None, num_prompts=64, sharegpt_output_len=None, sharegpt_context_len=None, random_input_len=16, random_output_len=32, random_range_ratio=1.0, image_count=1, image_resolution='480x720', random_image_count=False, image_format='jpeg', image_content='random', request_rate=inf, use_trace_timestamps=False, max_concurrency=None, output_file=None, output_details=False, print_requests=False, disable_tqdm=False, disable_stream=False, return_logprob=False, top_logprobs_num=0, token_ids_logprob=None, logprob_start_len=-1, return_routed_experts=False, seed=1, disable_ignore_eos=False, temperature=0.0, top_p=1.0, extra_request_body=None, apply_chat_template=True, profile=False, plot_throughput=False, profile_activities=['CPU', 'GPU'], profile_start_step=None, profile_steps=None, profile_num_steps=None, profile_by_stage=False, profile_stages=None, profile_output_dir=None, profile_prefix=None, lora_name=None, lora_request_distribution='uniform', lora_zipf_alpha=1.5, prompt_suffix='', pd_separated=False, profile_prefill_url=None, profile_decode_url=None, flush_cache=False, warmup_requests=1, tokenize_prompt=False, gsp_num_groups=64, gsp_prompts_per_group=16, gsp_system_prompt_len=2048, gsp_question_len=128, gsp_output_len=256, gsp_range_ratio=1.0, gsp_fast_prepare=False, gsp_send_routing_key=False, gsp_num_turns=1, gsp_ordered=False, gsp_group_distribution='uniform', gsp_zipf_alpha=None, mooncake_slowdown_factor=1.0, mooncake_num_rounds=1, mooncake_workload='conversation', fake_prefill=False, tag=None, header=None)

#Input tokens: 22980
#Output tokens: 2048
#Total images: 64
#Images per request: 1 (fixed)

=== Token Breakdown (per request avg / total) ===
  Raw text prompt tokens (without overhead): avg=16.0, total=1024
  Text prompt tokens (with chat template): avg=27.5, total=1757
  Text prompt overhead: avg=11.5, total=733
  Vision tokens: avg=331.6, total=21223

Created 64 random jpeg images with average 348518 bytes per request
Starting warmup with 1 sequences...
Warmup completed with 1 sequences. Starting main benchmark run...

============ Serving Benchmark Result ============
Backend:                                 sglang-oai-chat
Traffic request rate:                    inf       
Max request concurrency:                 not set   
Successful requests:                     64        
Benchmark duration (s):                  5.19      
Total input tokens:                      22980     
Total input text tokens:                 1757      
Total input vision tokens:               21223     
Total generated tokens:                  2048      
Total generated tokens (retokenized):    2048      
Request throughput (req/s):              12.34     
Input token throughput (tok/s):          4430.91   
Output token throughput (tok/s):         394.89    
Peak output token throughput (tok/s):    1126.00   
Peak concurrent requests:                64        
Total token throughput (tok/s):          4825.80   
Concurrency:                             63.51     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   5146.47   
Median E2E Latency (ms):                 5152.07   
P90 E2E Latency (ms):                    5168.12   
P99 E2E Latency (ms):                    5172.89   
---------------Time to First Token----------------
Mean TTFT (ms):                          3789.22   
Median TTFT (ms):                        4291.60   
P99 TTFT (ms):                           4743.64   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          43.78     
Median TPOT (ms):                        27.53     
P99 TPOT (ms):                           113.93    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           43.78     
Median ITL (ms):                         12.60     
P95 ITL (ms):                            16.44     
P99 ITL (ms):                            1790.03   
Max ITL (ms):                            3988.73   
==================================================

            ```

            ## Workload Stderr Tail

            ```text

  0%|          | 0/64 [00:00<?, ?it/s]
  2%|▏         | 1/64 [00:05<05:26,  5.17s/it]
100%|██████████| 64/64 [00:05<00:00, 12.35it/s]

            ```
