# Enhanced Free LLM API Resources

A comprehensive and well-organized list of free LLM inference resources accessible via API, with detailed parameters and filtering options.

## Table of Contents
- [Free Providers](#free-providers)
- [Providers with Trial Credits](#providers-with-trial-credits)
- [Filtering Options](#filtering-options)
- [Contributing](#contributing)
- [License](#license)

## Free Providers

### OpenRouter
**Limits/Notes**: 20 requests/minute, 200 requests/day

| Model Name | Model Limits |
|------------|--------------|
| DeepHermes 3 Llama 3 8B Preview | - |
| DeepSeek R1 | - |
| DeepSeek R1 Distill Llama 70B | - |
| DeepSeek R1 Distill Qwen 14B | - |
| DeepSeek R1 Distill Qwen 32B | - |
| DeepSeek R1 Zero | - |
| DeepSeek V3 | - |
| Dolphin 3.0 Mistral 24B | - |
| Dolphin 3.0 R1 Mistral 24B | - |
| Gemini 2.0 Flash Lite Preview 02-05 | - |
| Gemma 2 9B Instruct | - |
| Gemma 3 12B Instruct | - |
| Gemma 3 1B Instruct | - |
| Gemma 3 27B Instruct | - |
| Gemma 3 4B Instruct | - |
| Llama 3 8B Instruct | - |
| Llama 3.1 8B Instruct | - |
| Llama 3.1 Nemotron 70B Instruct | - |
| Llama 3.2 11B Vision Instruct | - |
| Llama 3.2 1B Instruct | - |
| Llama 3.2 3B Instruct | - |
| Llama 3.3 70B Instruct | - |
| Mistral 7B Instruct | - |
| Mistral Nemo | - |
| Mistral Small 24B Instruct 2501 | - |
| Moonlight-16B-A3B-Instruct | - |
| Mythomax L2 13B | - |
| OlympicCoder 32B | - |
| OlympicCoder 7B | - |
| OpenChat 7B | - |
| Phi-3 Medium 128k Instruct | - |
| Phi-3 Mini 128k Instruct | - |
| Qwen 2 7B Instruct | - |
| Qwen 2.5 72B Instruct | - |
| Qwen QwQ 32B | - |
| Qwen QwQ 32B Preview | - |
| Qwen2.5 Coder 32B Instruct | - |
| Qwen2.5 VL 72B Instruct | - |
| Reka Flash 3 | - |
| Rogue Rose 103B v0.2 | - |
| Toppy M 7B | - |
| Zephyr 7B Beta | - |
| mistralai/mistral-small-3.1-24b-instruct:free | - |

### Google AI Studio
**Limits/Notes**: Data is used for training (when used outside of the UK/CH/EEA/EU)

| Model Name | Model Limits |
|------------|--------------|
| Gemini 2.0 Flash | 1,000,000 tokens/minute, 1,500 requests/day, 15 requests/minute |
| Gemini 2.0 Flash-Lite | 1,000,000 tokens/minute, 1,500 requests/day, 30 requests/minute |
| Gemini 2.0 Flash (Experimental) | 4,000,000 tokens/minute, 1,500 requests/day, 10 requests/minute |
| Gemini 2.0 Pro (Experimental) | 5,000,000 tokens/day, 2,000,000 tokens/minute, 50 requests/day, 2 requests/minute |
| Gemini 1.5 Flash | 1,000,000 tokens/minute, 1,500 requests/day, 15 requests/minute |
| Gemini 1.5 Flash-8B | 1,000,000 tokens/minute, 1,500 requests/day, 15 requests/minute |
| Gemini 1.5 Pro | 32,000 tokens/minute, 50 requests/day, 2 requests/minute |
| LearnLM 1.5 Pro (Experimental) | 1,500 requests/day, 15 requests/minute |
| Gemma 3 27B Instruct | 15,000 tokens/minute, 14,400 requests/day, 30 requests/minute |
| text-embedding-004 | 150 batch requests/minute, 1,500 requests/minute, 100 content/batch |
| embedding-001 | - |

### Mistral (La Plateforme)
**Limits/Notes**: Free tier (Experiment plan) requires opting into data training, requires phone number verification

| Model Name | Model Limits |
|------------|--------------|
| Open and Proprietary Mistral models | 1 request/second, 500,000 tokens/minute, 1,000,000,000 tokens/month |

### Mistral (Codestral)
**Limits/Notes**: Currently free to use, monthly subscription based, requires phone number verification

| Model Name | Model Limits |
|------------|--------------|
| Codestral | 30 requests/minute, 2,000 requests/day |

### HuggingFace Serverless Inference
**Limits/Notes**: Limited to models smaller than 10GB. Some popular models are supported even if they exceed 10GB

| Model Name | Model Limits |
|------------|--------------|
| Various open models | Variable credits per month, currently $0.10 |

### Cerebras
**Limits/Notes**: Free tier restricted to 8K context

| Model Name | Model Limits |
|------------|--------------|
| Llama 3.1 8B | 30 requests/minute, 60,000 tokens/minute, 900 requests/hour, 1,000,000 tokens/hour, 14,400 requests/day, 1,000,000 tokens/day |
| Llama 3.3 70B | 30 requests/minute, 60,000 tokens/minute, 900 requests/hour, 1,000,000 tokens/hour, 14,400 requests/day, 1,000,000 tokens/day |

### Groq
**Limits/Notes**: -

| Model Name | Model Limits |
|------------|--------------|
| Allam 2 7B | 7,000 requests/day, 6,000 tokens/minute |
| DeepSeek R1 Distill Llama 70B | 1,000 requests/day, 6,000 tokens/minute |
| DeepSeek R1 Distill Qwen 32B | 1,000 requests/day, 6,000 tokens/minute |
| Distil Whisper Large v3 | 7,200 audio-seconds/minute, 2,000 requests/day |
| Gemma 2 9B Instruct | 14,400 requests/day, 15,000 tokens/minute |
| Llama 3 70B | 14,400 requests/day, 6,000 tokens/minute |
| Llama 3 8B | 14,400 requests/day, 6,000 tokens/minute |
| Llama 3.1 8B | 14,400 requests/day, 6,000 tokens/minute |
| Llama 3.2 11B Vision | 7,000 requests/day, 7,000 tokens/minute |
| Llama 3.2 1B | 7,000 requests/day, 7,000 tokens/minute |
| Llama 3.2 3B | 7,000 requests/day, 7,000 tokens/minute |
| Llama 3.2 90B Vision | 3,500 requests/day, 7,000 tokens/minute |
| Llama 3.3 70B | 1,000 requests/day, 6,000 tokens/minute |
| Llama 3.3 70B (Speculative Decoding) | 1,000 requests/day, 6,000 tokens/minute |
| Llama Guard 3 8B | 14,400 requests/day, 15,000 tokens/minute |
| Mistral Saba 24B | 1,000 requests/day, 6,000 tokens/minute |
| Qwen 2.5 32B | 1,000 requests/day, 6,000 tokens/minute |
| Qwen 2.5 Coder 32B | 1,000 requests/day, 6,000 tokens/minute |
| Qwen QwQ 32B | 1,000 requests/day, 6,000 tokens/minute |
| Whisper Large v3 | 7,200 audio-seconds/minute, 2,000 requests/day |
| Whisper Large v3 Turbo | 7,200 audio-seconds/minute, 2,000 requests/day |

### OVH AI Endpoints (Free Beta)
**Limits/Notes**: -

| Model Name | Model Limits |
|------------|--------------|
| Codestral Mamba 7B v0.1 | 12 requests/minute |
| DeepSeek R1 Distill Llama 70B | 12 requests/minute |
| Llama 3.1 70B Instruct | 12 requests/minute |
| Llama 3.1 8B Instruct | 12 requests/minute |
| Llama 3.3 70B Instruct | 12 requests/minute |
| Llava Next Mistral 7B | 12 requests/minute |
| Mistral 7B Instruct v0.3 | 12 requests/minute |
| Mistral Nemo 2407 | 12 requests/minute |
| Mixtral 8x7B Instruct | 12 requests/minute |

### Together
**Limits/Notes**: -

| Model Name | Model Limits |
|------------|--------------|
| Llama 3.2 11B Vision Instruct | - |
| Llama 3.3 70B Instruct | - |
| DeepSeek R1 Distil Llama 70B | - |

### Cohere
**Limits/Notes**: 20 requests/min, 1,000 requests/month

| Model Name | Model Limits |
|------------|--------------|
| Command-R | Shared Limit |
| Command-R+ | Shared Limit |
| Command-A | Shared Limit |

### GitHub Models
**Limits/Notes**: Extremely restrictive input/output token limits. Rate limits dependent on Copilot subscription tier (Free/Pro/Business/Enterprise)

| Model Name | Model Limits |
|------------|--------------|
| AI21 Jamba 1.5 Large | - |
| AI21 Jamba 1.5 Mini | - |
| Codestral 25.01 | - |
| Cohere Command R | - |
| Cohere Command R 08-2024 | - |
| Cohere Command R+ | - |
| Cohere Command R+ 08-2024 | - |
| Cohere Embed v3 English | - |
| Cohere Embed v3 Multilingual | - |
| DeepSeek-R1 | - |
| DeepSeek-V3 | - |
| JAIS 30b Chat | - |
| Llama-3.2-11B-Vision-Instruct | - |
| Llama-3.2-90B-Vision-Instruct | - |
| Llama-3.3-70B-Instruct | - |
| Meta-Llama-3-70B-Instruct | - |
| Meta-Llama-3-8B-Instruct | - |
| Meta-Llama-3.1-405B-Instruct | - |
| Meta-Llama-3.1-70B-Instruct | - |
| Meta-Llama-3.1-8B-Instruct | - |
| Ministral 3B | - |
| Mistral Large | - |
| Mistral Large (2407) | - |
| Mistral Large 24.11 | - |
| Mistral Nemo | - |
| Mistral Small | - |
| Mistral Small 3.1 | - |
| OpenAI GPT-4o | - |
| OpenAI GPT-4o mini | - |
| OpenAI Text Embedding 3 (large) | - |
| OpenAI Text Embedding 3 (small) | - |
| OpenAI o1 | - |
| OpenAI o1-mini | - |
| OpenAI o1-preview | - |
| OpenAI o3-mini | - |
| Phi-3-medium instruct (128k) | - |
| Phi-3-medium instruct (4k) | - |
| Phi-3-mini instruct (128k) | - |
| Phi-3-mini instruct (4k) | - |
| Phi-3-small instruct (128k) | - |
| Phi-3-small instruct (8k) | - |
| Phi-3.5-MoE instruct (128k) | - |
| Phi-3.5-mini instruct (128k) | - |
| Phi-3.5-vision instruct (128k) | - |
| Phi-4 | - |
| Phi-4-mini-instruct | - |
| Phi-4-multimodal-instruct | - |

### Chutes
**Limits/Notes**: Distributed, decentralized crypto-based compute. Data is sent to individual hosts

| Model Name | Model Limits |
|------------|--------------|
| Various open models | - |

### Cloudflare Workers AI
**Limits/Notes**: 10,000 neurons/day

| Model Name | Model Limits |
|------------|--------------|
| DeepSeek R1 Distill Qwen 32B | - |
| Deepseek Coder 6.7B Base (AWQ) | - |
| Deepseek Coder 6.7B Instruct (AWQ) | - |
| Deepseek Math 7B Instruct | - |
| Discolm German 7B v1 (AWQ) | - |
| Falcom 7B Instruct | - |
| Gemma 2B Instruct (LoRA) | - |
| Gemma 7B Instruct | - |
| Gemma 7B Instruct (LoRA) | - |
| Hermes 2 Pro Mistral 7B | - |
| Llama 2 13B Chat (AWQ) | - |
| Llama 2 7B Chat (FP16) | - |
| Llama 2 7B Chat (INT8) | - |
| Llama 2 7B Chat (LoRA) | - |
| Llama 3 8B Instruct | - |
| Llama 3 8B Instruct | - |
| Llama 3 8B Instruct (AWQ) | - |
| Llama 3.1 8B Instruct | - |
| Llama 3.1 8B Instruct (AWQ) | - |
| Llama 3.1 8B Instruct (FP8) | - |
| Llama 3.2 11B Vision Instruct | - |
| Llama 3.2 1B Instruct | - |
| Llama 3.2 3B Instruct | - |
| Llama 3.3 70B Instruct (FP8) | - |
| Llama Guard 3 8B | - |
| LlamaGuard 7B (AWQ) | - |
| Mistral 7B Instruct v0.1 | - |
| Mistral 7B Instruct v0.1 (AWQ) | - |
| Mistral 7B Instruct v0.2 | - |
| Mistral 7B Instruct v0.2 (LoRA) | - |
| Neural Chat 7B v3.1 (AWQ) | - |
| OpenChat 3.5 0106 | - |
| OpenHermes 2.5 Mistral 7B (AWQ) | - |
| Phi-2 | - |
| Qwen 1.5 0.5B Chat | - |
| Qwen 1.5 1.8B Chat | - |
| Qwen 1.5 14B Chat (AWQ) | - |
| Qwen 1.5 7B Chat (AWQ) | - |
| SQLCoder 7B 2 | - |
| Starling LM 7B Beta | - |
| TinyLlama 1.1B Chat v1.0 | - |
| Una Cybertron 7B v2 (BF16) | - |
| Zephyr 7B Beta (AWQ) | - |

### Google Cloud Vertex AI
**Limits/Notes**: Very stringent payment verification for Google Cloud

| Model Name | Model Limits |
|------------|--------------|
| Llama 3.1 70B Instruct | Llama 3.1 API Service free during preview. 60 requests/minute |
| Llama 3.1 8B Instruct | Llama 3.1 API Service free during preview. 60 requests/minute |
| Llama 3.2 90B Vision Instruct | Llama 3.2 API Service free during preview. 30 requests/minute |
| Llama 3.3 70B Instruct | Llama 3.3 API Service free during preview. 30 requests/minute |
| Gemini 2.0 Flash Experimental | Experimental Gemini model. 10 requests/minute |
| Gemini 2.0 Flash Thinking Experimental | - |
| Gemini 2.0 Pro Experimental | - |

## Providers with Trial Credits

### Together
**Credits**: $1 when you add a payment method

| Model Types | Requirements |
|-------------|--------------|
| Various open models | - |

### Fireworks
**Credits**: $1

| Model Types | Requirements |
|-------------|--------------|
| Various open models | - |

### Unify
**Credits**: $5 when you add a payment method

| Model Types | Requirements |
|-------------|--------------|
| Routes to other providers, various open models and proprietary models (OpenAI, Gemini, Anthropic, Mistral, Perplexity, etc) | - |

### NVIDIA NIM
**Credits**: 1,000 API calls for 1 month

| Model Types | Requirements |
|-------------|--------------|
| Various open models | - |

### Baseten
**Credits**: $30

| Model Types | Requirements |
|-------------|--------------|
| Any supported model - pay by compute time | - |

### Nebius
**Credits**: $1

| Model Types | Requirements |
|-------------|--------------|
| Various open models | - |

### Novita
**Credits**: $0.5 for 1 year, $20 for 3 months for DeepSeek models with referral code + GitHub account connection

| Model Types | Requirements |
|-------------|--------------|
| Various open models | - |

### Hyperbolic
**Credits**: $1

| Model List | Requirements |
|------------|--------------|
| DeepSeek V3, Hermes 3 Llama 3.1 70B, Llama 3 70B Instruct, Llama 3.1 405B Base, Llama 3.1 405B Base (FP8), Llama 3.1 405B Instruct, Llama 3.1 70B Instruct, Llama 3.1 8B Instruct, Llama 3.2 3B Instruct, Llama 3.3 70B Instruct, Pixtral 12B (2409), Qwen QwQ 32B, Qwen QwQ 32B Preview, Qwen2.5 72B Instruct, Qwen2.5 Coder 32B Instruct, Qwen2.5 VL 72B Instruct, Qwen2.5 VL 7B Instruct | - |

### SambaNova Cloud
**Credits**: $5 for 3 months

| Model List | Requirements |
|------------|--------------|
| Llama 3.1 405B, Llama 3.1 70B, Llama 3.1 8B, Llama 3.2 11B Vision, Llama 3.2 1B, Llama 3.2 3B, Llama 3.2 90B Vision, Llama 3.3 70B, Llama-Guard-3-8B, Qwen/QwQ-32B, Qwen/QwQ-32B-Preview, Qwen/Qwen2-Audio-7B-Instruct, Qwen/Qwen2.5-72B-Instruct, Qwen/Qwen2.5-Coder-32B-Instruct, allenai/Llama-3.1-Tulu-3-405B, deepseek-ai/DeepSeek-R1, deepseek-ai/DeepSeek-R1-Distill-Llama-70B, tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3, tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3 | - |

### Scaleway Generative APIs
**Credits**: 1,000,000 free tokens

| Model List | Requirements |
|------------|--------------|
| BGE-Multilingual-Gemma2, DeepSeek R1 Distill Llama 70B, DeepSeek R1 Distill Llama 8B, Llama 3.1 70B Instruct, Llama 3.1 8B Instruct, Llama 3.3 70B Instruct, Mistral Nemo 2407, Pixtral 12B (2409), Qwen2.5 Coder 32B Instruct, sentence-t5-xxl | - |

### AI21
**Credits**: $10 for 3 months

| Model Types | Requirements |
|-------------|--------------|
| Jamba/Jurrasic-2 | - |

### Upstage
**Credits**: $10 for 3 months

| Model Types | Requirements |
|-------------|--------------|
| Solar Pro/Mini | - |

### NLP Cloud
**Credits**: $15

| Model Types | Requirements |
|-------------|--------------|
| Various open models | Phone number verification |

### Alibaba Cloud (International) Model Studio
**Credits**: Token/time-limited trials on a per-model basis

| Model Types | Requirements |
|-------------|--------------|
| Various open and proprietary Qwen models | - |

### Modal
**Credits**: $30/month

| Model Types | Requirements |
|-------------|--------------|
| Any supported model - pay by compute time | - |

## Filtering Options

### By Model Type
- Text Generation
- Chat
- Code Generation
- Image Generation
- Embeddings
- Vision Models
- Audio Models

### By API Type
- REST
- WebSocket
- GraphQL

### By Features
- Streaming
- Function Calling
- Fine-tuning
- Custom Models
- Multi-modal
- Vision Capabilities
- Audio Processing

### By Rate Limits
- Daily Limits
- Monthly Limits
- Pay-as-you-go
- Request-based Limits
- Token-based Limits

## Additional Parameters

Each provider entry includes:
- Authentication Method
- Supported Languages
- Response Format
- Error Handling
- Rate Limiting Details
- Pricing (if applicable)
- Example Usage
- Best Practices

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

Please use these resources responsibly and in accordance with each provider's terms of service. Some providers may require registration or have specific usage restrictions. 
