# 接触模型备份：smooth_positive

本分支在测量锚定增量弹簧实现之前保存。包含当前项目源代码、场景、网格、绘图和诊断代码。
手持命令行的接触模型默认值为 `smooth_positive`；平放场景仍保留当时的 smooth_positive 默认。
两种旧法则都在共享核心中，尚未包含 measured_anchored_ideal。

平滑版：kx * 0.5 * (c + sqrt(c*c + eps*eps)) + bias。
旧理想版：kx * max(0, c) + bias。bias 均在截断之外。
这些是历史备份，不表示旧理想版本通过闭环验收；其曾新增触限、误差和短暂单侧失接触。
同目录保留当时对账结果。工作树原始字节哈希会因 Git 文本换行规范化而与检出字节不同。
生成的求解器 DLL、输出日志和大体积历史 ZIP 不上传；本地完整历史备份仍保留在
`D:/RobotSim/_archive_20261012/ideal_contact_full_backup/`。
本次额外本地源码备份：`D:/RobotSim/_archive_20261012/measured_anchored_contact_before/source_before.zip`。

Python 使用 D:/acados_env/python.exe。模型 shared_v2；首次缺缓存时重新编译。
