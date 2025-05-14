Скрипт для запуска - `devkaZNH.sh`

Запускается путем вызова файла в терминале соответственно.

В скрипте указываем все необходимые для метода параметры, оптимизатор указываем в строке `command+=" --optimizer aid_with_adam"`. 

Файл с самим обучением - `torchrun_main.py`. В нем считываются все указанные параметры и вызывается указанный оптимизатор. 

| Метод           | Название оптимизатора в скрипте | Название класса          | Файл                        |
|-----------------|-----------------------|--------------------------|-----------------------------|
| ALIAS           | aid_with_adam                 | `AIDWithAdam`            | `parameter_free_signsgd.py` |
| SignSGD         | sgd_with_adam               | `SgdWithAdam`            | `sgd_with_adam.py`          |
| SteepestDescent | sgd_with_adam      | `SgdWithAdam` (с флагами `--sign True` и `--sign_norm True`) | `sgd_with_adam.py` |
| ADAM-LIKE       | adam_like           | `AdamLike`               | `parameter_free_signsgd.py` |
| Prodigy         | prodigy               | `ProdigyWithAdam`        | `parameter_free_signsgd.py` |
| ADAMW           | adamw                 | (используется напрямую `torch.optim.AdamW`) | - |

Параметры типа `cosine` и `constant lr` указываются в скрипте запуска в `--scheduler`. 
