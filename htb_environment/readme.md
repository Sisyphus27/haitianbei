# htb_environment

- pipeline/pipeline.py：知识图谱→先验知识→RL→生成调度计划→更新kg→下一轮..

  1. 创建并连接任务一的 KG 客户端 Dataset_KG

  2. 用 KG 构造先验适配器 T1KGPriorAdapter，并 attach 到环境

  3. rollout -> buffer -> agent.train

  4. 选出本 epoch 最优/最后一组调度，转三元组 -> 调用任务一接口 update_with_triples 回写

  5. 下一 epoch 自动使用更新后的图谱统计
- pipeline/kg_bridge.py：生成先验（plane_prior、site_prior，在environment.py attach_prior()），调度转换为三元组方便更新
- environment.py：仿真环境
- main.py：主流程入口，默认带有--use_task1_kg调用pipeline，若无该参数，只进行调度

- utils/schedule_converter.py：绘图工具类
- MARL/arguments.py：各类参数设置
- MARL/runner.py：调度算法逻辑
