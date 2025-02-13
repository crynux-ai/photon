

M1 Max, 32 GPU core, 64GB Mem.


|           | FP32PP   | FP32TG | F16PP | F16TG  | Q8PP  | Q8TG | Q4PP  | Q4TG |
|-----------|----------|--------|-------|--------|-------|------|-------|------|
|Llama.cpp[^1]  |      |        | 599.53|  23.03 |537.37 | 40.2 | 530.06| 61.19|
|MLX [^2]   |          |        | 652   |  19    |       |      |  438  |   31 |
|Pytorch CPU| 59.83    |  1.01  |       |        |       |      |       |      |

[^1]: https://github.com/ggerganov/llama.cpp/discussions/4167
[^2]: https://medium.com/@andreask_75652/benchmarking-apples-mlx-vs-llama-cpp-bbbebdc18416