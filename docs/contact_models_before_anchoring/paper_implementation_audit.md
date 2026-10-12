# Zhao 2024 与当前双臂手持实现：逐项证据审计

**版本界定：本文审计对象是已测试理想候选的冻结 source，17 项原始 SHA256 全部保留并已逐一校验。因闭环验收出现新增触限及部分误差增大，最终生产入口已恢复 smooth_positive 默认，ideal_unilateral 仅为显式实验选项；后文“新手持模型”指候选，不代表最终默认。代码链接固定到已校验的候选快照；候选未打包的平放 exp2_slender_scene.py 另存 paper_source_snapshot，acados 外部库仍指原路径。原理想候选失败结果未删除。**

当前程序保留了论文的混合协作误差与非线性滚动优化思路，但不是论文模型的逐公式复现。优先差异是：**E/epsilon代价层级、终端输入项、冻结动力学与缺显式Coriolis、额外kf缩放、冻结bias/接触残差及世界x接触近似**。新理想单向弹簧仅替换了名义力曲线，没有消除其余近似。

- 论文：Xingwei Zhao、Yuhao Zhang、Wenzheng Ding、Bo Tao（通讯）、Han Ding，*A Dual-Arm Robot Cooperation Framework Based on a Nonlinear Model Predictive Cooperative Control*，IEEE/ASME Transactions on Mechatronics 29(5), 3993–4005, 2024；[DOI](https://doi.org/10.1109/TMECH.2023.3263357)。
- 作者所属单位：华中科技大学机械学院；[作者张宇豪页面](https://mse.hust.edu.cn/info/1068/15991.htm)列有此论文；[实验室](https://imet.hust.edu.cn/)。
- 论文类型：方法与实体实验，辅以动力学/静态关系推导。任务范围是针对实现的审计，报告采用用户要求的Markdown证据表。
- 已读PDF全部13页，包括附录、参考文献与作者页；式12/31/32另以页面图目视确认。下文“p”指PDF页，印刷页=3992+p。
- 审计代码：已测试的 `candidate_cop08_f02_move05/source` 理想候选快照（含 shared_v2）；下表行号及 SHA256 固定对应此快照。`experiments/sampling.py`不存在，真实采样位于`diagnostics/measurements.py`，调用方为`experiments/runner.py`。精确源码SHA256与片段记录于同目录`paper_audit_evidence.json`。
- 分类：“基本对应”不等于逐值相同；“未说明”不等于bug；“字面差异”保留作者意图未知；“推断”只解释可能影响，不作为实测因果。

## 完整证据表

### 目标与协作误差

| ID/主题 | 论文明确内容 | 代码采用与精确锚点 | 判定、影响与边界 |
|---|---|---|---|
| A01 组合误差代数和符号 | p3,4 / 印刷3995,3996，11–12：etrac=ri-rref；ecop=rj-ri；E=(I-eta_f)[(I-eta_cop)etrac-eta_cop ecop]-eta_f ef。 | 实现 (1-eta_f)*(ri-(1-eta_cop)rref-eta_cop*rj)-eta_f*kf*(Fi-Fref)。除kf外，与展开后的E一致。 [model.py:57–64](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:57) | **基本对应，但另有力缩放**。左右符号及协作项未发现反号证据。位移项与力项可相互抵消，E接近零不等于独立位置/力误差都小；这与论文混合误差的结构一致。 限制：不凭代数一致证明闭环稳定或全部坐标约定等效。 |
| A02 额外 kf_scale | p4,10 / 印刷3996,4002，12,16,38：这些公式直接混合位置误差与力误差，未给独立kf乘子或单位归一化表。 | 两场景默认kf_scale=.003，运行时参数p[188]，直接乘力误差和力率误差。 [model.py:61–64](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:61)；[parameters.py:63–68](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/parameters.py:63)；[handheld_cli.py:10–24](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scenes/handheld_cli.py:10) | **论文未说明的建模选择**。同一eta_f不能脱离kf直接与论文对比。它是位力单位/灵敏度选择，会改变静态妥协；不是由论文给出或由kx自动确定。 限制：论文可能有未披露归一化，官方源码未获取；不认定.003本身错误，也不擅自改为1。 |
| A03 E 与 epsilon 的代价层级 | p4,5,6 / 印刷3996,3997,3998，12,31,32：Eq12明确epsilon_i=E_i^T E_i；Eq31/32使用epsilon向量及其导数的二次项。若逐字代入，位置项为(E_i^T E_i)^2，导数项涉及(2E_i^T Edot_i)^2。 | residual直接拼接两侧6维E及6维Edot，然后NONLINEAR_LS二次加权；不是对每臂标量epsilon再平方。 [model.py:57–66](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:57)；[ocp.py:46–54](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/ocp.py:46)；[ocp.py:84–86](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/ocp.py:84) | **明确字面差异；作者意图未决**。当前是E的二次LS。它与论文式12、31、32连用的字面四次目标不同，连梯度/远近误差权衡都可能改变；须显式声明。 限制：论文也称quadratic terms，可能符号复用/排版错误。需作者或官方源码澄清；不能据此直接将当前实现改四次。已目视核验p4–6公式，非OCR误读。 |
| A04 逐轴、双臂对称系数 | p4,9,10 / 印刷3996,4001,4002，12; Experiment II; Table IV：eta是逐轴对角矩阵。实验II x力系数.8，y协作系数.8，两臂相同；论文也讨论不对称系数。 | scene.tile使两臂eta相同，仅y协作和x力可由CLI调，其余为0；公共参数层可接收(2,6)。 [scene.py:104–119](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scene.py:104)；[parameters.py:67–68](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/parameters.py:67) | **与论文实验II任务分工一致**。默认任务方向与论文一致；当前CLI是框架的对称子集。00不等于无夹持命令：x位置参考仍预压、动力学仍包含接触。 限制：改变eta的六/八组研究是本项目实验，不是论文复现中原样给出的扫描。 |
| A05 位姿坐标与相对物体偏置 | p2,3 / 印刷3994,3995，1,2,11：位姿为xyz及三个旋转分量；Eq2有base和工件变换，随后为简洁将固定工件向量并入另一臂。未指定旋转向量主值误差。 | FK输出世界xyz+主值rotvec，直接分量相减；协作项用两tip原坐标，无一般动态工件SE(3)变换。默认只y协作，因此无需把x夹持间距直接压成零。 [kinematics.py:9–14](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/kinematics.py:9)；[kinematics.py:120–134](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/kinematics.py:120)；[model.py:61–64](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:61)；[scene.py:108–109](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scene.py:108) | **实现约定/适用范围限制**。大旋转/一般不对称任务不能无条件沿用当前误差；角度rad与平移m进入同一对角LS也没有论文中明确的标度。 限制：本次小姿态变化是否产生显著误差需单独验证；未发现当前运行已跨主值分支的证据。 |

### 预测动力学

| ID/主题 | 论文明确内容 | 代码采用与精确锚点 | 判定、影响与边界 |
|---|---|---|---|
| B01 状态空间和质量矩阵 | p2,6,7,8 / 印刷3994,3998,3999,4000，1,25,35：一般推导用笛卡尔位姿/速度与状态相关M；两连杆例用关节动力学M(q)。实验可从Panda读取质量和重力矩阵。 | 优化状态28维=[qL,dqL,qR,dqR]，输入14关节力矩，FK非线性；从MuJoCo完整质量矩阵取两块7x7，每拍复制到所有N+1节点。 [model.py:26–35](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:26)；[measurements.py:74–87](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/diagnostics/measurements.py:74)；[parameters.py:54–59](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/parameters.py:54) | **工程离散化与冻结近似**。改用关节状态本身不构成错误，但M随预测q不再更新是明确近似。当前质量是在恢复真实速度并forward后取，不是故意用上一步缓存M。 限制：两臂间/箱体耦合由接触解决，预测没有完整箱体状态或刚性闭链约束；不能声称重建论文完整连续模型。 |
| B02 重力、Coriolis与阻尼 | p2,6,7,8 / 印刷3994,3998,3999,4000，1,25,35; §V setup：Eq25包含C(r,rdot)、G(r)；Eq35有C(q,qdot)。论文仿真及实验明确将难识别关节摩擦作为不确定项。 | 置qvel=0再forward读取qfrc_bias作为重力，恢复速度后未另取速度相关bias；模型只有冻结G和damping*dq，场景阻尼设.5。 [measurements.py:72–87](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/diagnostics/measurements.py:72)；[model.py:48–50](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:48)；[scene.py:81–82](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scene.py:81) | **明确模型简化**。Coriolis/离心项没有显式进入预测；冻结G也是近似。线性粘性阻尼不是Coriolis替代。论文允许未知摩擦不等于证明可忽略所有速度相关动力学。 限制：未评估该近似在本轨迹中的量级；不能直接归因保持力偏差或触限。 |
| B03 完整接触残差 | p5,6 / 印刷3997,3998，25–27：原文以J^T/J^-T映射相互作用力，没有给当前全接触载荷减法向模型并冻结的算法。 | 当前模型法向载荷回算后，load=measured_joint_load-model_normal；所有节点共享该残差。tangent备选仅去掉世界x后映射。 [controller.py:79–98](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/controller.py:79)；[parameters.py:59–64](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/parameters.py:59)；[measurements.py:53–59](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/diagnostics/measurements.py:53) | **本项目测量补偿，不是原文给定**。当前节点保证model_normal+residual=measured_joint_load，避免当前法向重复计入；未来节点沿用当前切向/力偶/误差，不预测摩擦变化、滑移或接触丢失。 限制：残差并非完整动力学扰动观测器；只来自所选box-pad接触，不自动吸收缺失Coriolis、其他碰撞或关节限位反力。 |
| B04 世界x法向与接触映射 | p2,5,6 / 印刷3994,3997,3998，1,25–27：原文用笛卡尔交互力及Jacobian，一般公式未限制为世界x。实验II力约束在x。 | 预测只把pad-face点Jacobian转置乘[Fx,0,0]；真实测量全载荷在每个实际接触点用jp/jr映射。 [model.py:34–47](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:34)；[measurements.py:44–59](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/diagnostics/measurements.py:44)；[kinematics.py:44–59](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/kinematics.py:44) | **任务特化近似**。世界x仅在夹持面基本对向x时可解释为法向。箱体/垫片倾斜后预测仍不跟随真实接触法线；残差只能在当前拍补齐。 限制：不是contact frame.T或geom顺序反号的证据，二者此前独立测量审查已验证。 |
| B05 物体、摩擦和掉箱预测 | p5,9 / 印刷3997,4001，27; Experiment II：原文物体变形力给简化弹簧；实验靠夹持力和摩擦托住箱体，未给完整摩擦锥OCP。 | 28状态仅双臂；无箱体位姿、速度、接触模式、摩擦锥、抗倾覆或最小夹持力约束。物理MuJoCo有箱体和摩擦。 [model.py:29–31](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:29)；[model.py:67–68](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:67)；[ocp.py:87–99](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/ocp.py:87)；[scene.py:54–62](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scene.py:54) | **缺少显式模型/约束**。求解成功、预测E小不能证明不滑/不掉/真实无触限。该边界既是当前实现限制，也不能从原论文实验概括成数学安全保证。 限制：必须用实际物理日志和接触几何验收；不是发现代码应该自动拥有论文未给的完整接触规划器。 |

### 接触弹簧与力测量

| ID/主题 | 论文明确内容 | 代码采用与精确锚点 | 判定、影响与边界 |
|---|---|---|---|
| C01 平滑接触律及新理想分支 | p5,7,8 / 印刷3997,3999,4000，26,27,34,36：原文为线性关系k*(环境位置-末端位置)或两臂变形关系；未写sqrt平滑、max单向开关、gap偏移或2mm epsilon。 | 新手持Fnom=kx*if_else(c>0,c,0)，c=rest_gap-(faceRx-faceLx)，c=0选零导数；flat/旧cfg仍.5*kx*(c+sqrt(c²+eps²))，eps=.002。 [model.py:36–44](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:36)；[config.py:25–62](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/config.py:25)；[scene.py:116–119](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scene.py:116) | **两者均非原文逐字模型**。ideal移除零附近平滑斜率减半和正力尾部，但它是单向接触工程改型，不能称恢复论文原型。flat的同类平滑仍在。 限制：新非光滑点的求解收敛须单独实测；本审计不跑闭环、不调整参数。 |
| C02 当前力锚定和分离后的总力 | p5,6 / 印刷3997,3998，26,27,25：未写每拍测量减名义值的恒定加性bias。 | bias=measured-Fnom(current)，Fpred=Fnom(predicted)+bias；bias跨N+1节点冻结，不乘接触开关、不钳制总力。 [controller.py:68–79](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/controller.py:68)；[parameters.py:64–64](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/parameters.py:64)；[model.py:42–56](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:42) | **重要保留近似**。当前力相等来自显式锚定，不能当预测模型精确证据。即使ideal的c<=0让Fnom=0，总预测力仍等于bias，可能非零；左右bias也可破坏名义作用反作用对称。 限制：保持bias是本次用户授权变更的范围约束；报告风险而不额外改为夹断/互补约束。 |
| C03 力率、阻尼与时变bias | p4,5,6 / 印刷3996,3997,3998，12,31,32：目标包含误差导数；弹簧表达式随位置变化。没有bias动态或阻尼接触模型细节。 | 只对当前选择的几何弹簧求时间导数；bias、kx、rest_gap视为常数。切向力/力矩预测值由bias固定，预测率为0。 [model.py:42–64](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:42)；[parameters.py:64–68](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/parameters.py:64) | **冻结导数假设**。新ideal的力值与率共用同一分支，未发现遗漏切换成新导数；它仍不预测速度/控制相关的接触反力变化。 限制：连续时间物理反力的全导数不等于该冻结几何导数。 |
| C04 法向幅值、signed Fx与6D wrench | p2,3,9 / 印刷3994,3995,4001，1,6; Experiment II：论文frob是作用于机器人笛卡尔交互力；实验II报告x向10N，未给MuJoCo合成口径。 | handheld控制用signed world Fx+legacy六轴bias；flat用左右带符号normal abs sum且只x bias。绘图FL/FR始终normal abs sum。 [controller.py:69–79](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/controller.py:69)；[measurements.py:44–60](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/diagnostics/measurements.py:44)；[recording.py:166–173](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/recording.py:166) | **口径差异，不能混称**。normal abs sum不是重复把左右计两次，但倾斜或切向力时不等于\|world Fx\|。legacy接触自身torque不含r×f，真实face wrench另行记录；全joint_load包含正确力臂。 限制：默认eta_f仅x非零，遗漏反馈力臂扭矩目前不直接进入旋转force误差；若扩展旋转力系数就需重新审查。 |
| C05 左右映射和时序 | p3,9 / 印刷3995,4001，11; Experiment II：论文左右为物理机器人编号，未规定本模型世界x侧命名或仿真采样时序。 | handheld将physical-right映射controller-left；geom1/2决定反力号，frame.T转世界；sample在施加新u之前，Euler物理反力在新u下计算。 [handheld.py:49–67](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scenes/handheld.py:49)；[measurements.py:35–47](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/diagnostics/measurements.py:35)；[runner.py:76–106](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/runner.py:76) | **已见一致实现；有时间口径限制**。日志同t的node0锚定与实际物理反力可因ctrl不同而不同；此前0时刻10.129N与22.931N差异已由同状态不同u离线复核解释，不能判测量错。 限制：此前报告是旧smooth的0.8/0.2运行证据，不能直接替代新ideal运行的验收。 |
| C06 刚度、rest_gap与软接触 | p5,8,9 / 印刷3997,4000,4001，27,36; Experiment II：论文说明刚度影响交互力，但未给手持2419.3、flat2793.166、MuJoCo solref/solimp/noslip标定。 | 手持face restgap在几何REST_GAP上减两pad半厚；kx可与初始化g.K_STIFF独立设置；物理接触solref=[.02,1], solimp前三=.999,.999,.001，摩擦.8/noslip3。 [scene.py:58–82](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scene.py:58)；[scene.py:92–106](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scene.py:92)；[handheld.py:8–14](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scenes/handheld.py:8)；[handheld.py:140–145](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scenes/handheld.py:140) | **标定/物理模型选择，原文未给**。预测弹簧刚度不是MuJoCo材料常数。旧标定目标位移、实际face gap、控制/速度和截距口径需区分；已有独立微扰报告支持raw kx局部量级而非通用值。 限制：本轮不重新运行会写XML的标定，不使用F/压缩倒推刚度；详见此前kx_audit。 |
| C07 中心间隙不等于真实接触开闭 | p5,9 / 印刷3997,4001，27; Experiment II：原文只给简化物体变形关系，没有将两个pad-face中心的世界x间距等同碰撞检测。 | compression只用两个面中心世界x之差，不读预测box姿态、接触点、contact.dist或contact count；真实测量遍历MuJoCo检测到的接触点。 [model.py:34–39](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:34)；[kinematics.py:44–59](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/kinematics.py:44)；[measurements.py:35–59](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/diagnostics/measurements.py:35) | **必须区分几何代理量与接触事件**。c<=0只表示这个一维面中心代理量不压缩。倾斜面可能仍以边/角接触；c>0也不能单独证明足够摩擦或夹持稳定。因此新horizon中c<=0且总力非零应同时核对bias与真实几何，而非直接判掉箱。 限制：总力在数学名义开关外非零是C02可确认的模型特性；它在具体物理状态是否不合理，须结合真实接触位置和箱体姿态判断。 |

### 轨迹、初始化与采样

| ID/主题 | 论文明确内容 | 代码采用与精确锚点 | 判定、影响与边界 |
|---|---|---|---|
| D01 阶段轨迹与导数 | p9,10 / 印刷4001,4002，Experiment II; Fig12：给出-y 10mm/s、x 10N及夹持后搬运叙述，未指定.5s ramp、1s hold、.2s加减速的多项式。 | ramp用3s²-2s³；搬运速度smoothstep并解析积分位置、解析给速度；力参考率同ramp解析导数；末端姿态参考固定。 [reference.py:7–18](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/reference.py:7)；[reference.py:33–77](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/reference.py:33) | **工程轨迹设计**。这些平滑是参考规划，与已替换的接触平滑不是同一项。5cm/.01m/s加减速段总时长5.2s符合积分面积。 限制：phase函数运动结束后仍标move，实际参考已停止；分阶段验收应使用move_end而非只看字符串。 |
| D02 启动几何预压与保持位置 | p9,10 / 印刷4001,4002，Experiment II holding description：描述先压紧达到preload再稳定搬运，没有给本文IK/二分标定或定时切换代码。 | handheld的press0来自实测预压标定返回target_fn(command_force)，最终press来自target_fn(args.force)。即使f_start==force，二者也可能不同，ramp仍插值位置。flat不使用此校准。 [scene.py:87–101](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scene.py:87)；[handheld.py:97–126](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scenes/handheld.py:97)；[reference.py:69–77](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/reference.py:69) | **可能影响启动的独立设计差异**。因此“10N到10N”不保证启动位置参考不动；需检查保存reference的press0/press，不能只看力参考标量。 限制：是代码可确认的可能机制；本审计不将其定为偏差唯一原因、不改初始姿态。 |
| D03 控制、预测、物理与日志时间网格 | p6,7,8,9 / 印刷3998,3999,4000,4001，§III-D; §V; Fig10c：Panda控制周期1ms、预测时域100ms，报告平均0.28ms、每循环少于1ms；离散N及积分器细节未全给。 | DT=.002，CONTROL_EVERY=3，即控制6ms；N10/Tp.1即预测节点10ms；ERK每区间2子步4级；普通日志每5物理步即10ms，handheld physics每2ms。 [runner.py:12–13](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/runner.py:12)；[runner.py:62–83](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/runner.py:62)；[ocp.py:82–83](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/ocp.py:82)；[config.py:32–54](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/config.py:32)；[recording.py:125–142](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/recording.py:125) | **明确不同于论文实时条件**。当前是不同多速率实现，不能沿用论文1kHz实时结论。求解CPU时长也不能直接与包括通信的硬件循环比较。 限制：不要求预测节点必须等于控制间隔；6ms与10ms不是自动bug，但热启动和物理验收应明确。 |
| D04 初始化、热启动与失败处理 | p6,7 / 印刷3998,3999，§III-D：论文提到初始化缓冲、更新/错误停止及生成式实时求解；未披露每拍节点热启动平移细节。 | 同次测量计算平衡种子，做两次不推进物理的warmup；warm状态/控制整段原样复用，不按6ms或10ms平移；失败可重建实例并再求一次。 [runner.py:25–47](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/runner.py:25)；[controller.py:123–132](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/controller.py:123)；[controller.py:176–201](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/controller.py:176)；[controller.py:210–212](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/controller.py:210) | **工程求解策略**。有限RTI下初值会影响结果；两次预热和未移位复用不是论文中确认过的原算法。模型平衡seed也不是物理系统平衡保证。 限制：本审计未测试另一热启动策略，不建议悄悄替换。 |

### OCP与数值求解

| ID/主题 | 论文明确内容 | 代码采用与精确锚点 | 判定、影响与边界 |
|---|---|---|---|
| E01 Q/P、导数与输入权重 | p5,6 / 印刷3997,3998，31,32：Q/P分别权衡过程与终端，eta_dev和eta_u权衡导数和输入。原文未给当前默认1/2/.05/5e-8。 | 阶段38维[E12,Edot12,u14]，终端24维[E12,Edot12]；W=Q*diag(1,eta_dev,eta_u)，We=P*diag(1,eta_dev)，不对u减重力seed。 [model.py:65–66](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:65)；[ocp.py:46–54](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/ocp.py:46)；[config.py:36–43](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/config.py:36) | **结构部分对应，默认值未获论文支持**。Q和P独立整体缩放符合描述；代码仅提供标量整体Q/P加对角eta权重，是论文矩阵表述的受限子集。具体权值及力标度是本项目设计。导数残差涉及A03的E/epsilon歧义。 限制：不能仅相同eta_cop/eta_f就声称相同控制器。 |
| E02 终端输入代价 | p6 / 印刷3998，32：式32终端括号内显式含u^T eta_u u。 | terminal_residual只有24维位置/导数，无u；终端权矩阵不含eta_u。 [model.py:65–66](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:65)；[ocp.py:53–54](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/ocp.py:53)；[ocp.py:76–86](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/ocp.py:76) | **明确与论文式32字面不同**。当前acados离散终端节点没有控制变量，采用了省略终端输入惩罚的实现选择。需声明与原文不同。 限制：原文未给离散终端u定义；不能简单把最后一段u加进去就宣称等价，须先明确原作者离散规则。 |
| E03 积分代价的时间缩放 | p5 / 印刷3997，30,31：过程代价是预测时域上的积分。 | 应用代码给time_steps；本机acados_template在cost_scaling未设置时使用append(time_steps,1)，因此阶段按节点时长、终端按1缩放。 [ocp.py:82–86](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/ocp.py:82)；[acados_ocp.py:1232–1236](D:/acados/interfaces/acados_template/acados_template/acados_ocp.py:1232) | **已核对，不是漏乘dt**。不能仅因cost_weights中没有dt就认定遗漏积分。Q/P比应结合时间缩放理解。 限制：该结论绑定本机库源码SHA256，不确认论文ACADO所用的离散积分规则。 |
| E04 速度、力矩、关节边界 | p5,8 / 印刷3997,4000，30; §V setup：式30规定末端速度/最大输入，实验文字最大torque写10N（原文单位如此）；没有给关节限位、z单独限速和松弛项。 | 当前速度仅xyz欧氏范数，加独立z界；力矩盒范围来自XML；每个预测状态节点/终端关节硬界。无物理限位安全裕度、无接触约束，非线性速度约束只设置路径项。 [model.py:67–68](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:67)；[ocp.py:87–99](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/ocp.py:87)；[scene.py:122–126](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scene.py:122) | **部分对应，另有工程约束**。这些是适应本体的扩展，不能拿纸面无违反当真实2ms轨迹无触限保证。关节界无裕度且模型近似可能导致实际触限。 限制：纸面r是6维而速度norm单位未明确；当前仅平移速度不能无证据称完整等价或明确错误。 |
| E05 RTI、QP与Hessian | p6,7 / 印刷3998,3999，§III-D：ACADO生成C++、单/多重打靶、Gauss-Newton、凝聚QP、qpOASES；引用实时迭代文献。 | acados SQP_RTI、PARTIAL_CONDENSING_HPIPM、Gauss-Newton；默认QP迭代15、NLP20配置、ERK2步4级，MIRROR及LM=.01。 [ocp.py:100–113](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/ocp.py:100)；[config.py:47–56](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/config.py:47) | **同类实时思路，具体算法不同**。Gauss-Newton与论文一致；库、QP、正则和积分器不是原文逐项复现。RTI单次更新不等于每拍求得全局或收敛最优解。 限制：论文未给所有容差/正则/N，不能把本机默认当作者参数；未审计库底层数值正确性。 |
| E06 指令clip与速率约束 | p5 / 印刷3997，30：论文有输入幅值限制，未披露输出裁剪或增量约束。 | OCP力矩硬盒+runner再np.clip到actuator范围，保存clip/saturation标记；公共SolveOptions支持第一阶段du_max，但runner默认未传u_prev/du_max。 [controller.py:139–159](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/controller.py:139)；[runner.py:88–99](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/runner.py:88)；[runner.py:151–155](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/runner.py:151) | **明示保护，不是隐蔽测量滤波**。若clip发生实际u不再完全等于优化u，应报告clip比例；默认没有额外力矩低通或增量平滑。 限制：不是所有clip都是错误；这里只核对启用路径，不改变保护逻辑。 |

### 其他平滑、裁剪与适用域

| ID/主题 | 论文明确内容 | 代码采用与精确锚点 | 判定、影响与边界 |
|---|---|---|---|
| F01 速度范数epsilon | p5 / 印刷3997，30：原文使用速度范数，不含平滑常数。 | sqrt(v_xyz·v_xyz+1e-10)，静止值为1e-5m/s。 [model.py:67–68](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/model.py:67) | **数值正则，影响极小但存在**。这是避免零范数不可导的数值选择，未随接触ideal替换删除。对.05m/s上界影响微小；vmax接近零时须留意。 限制：量级为解析计算；本审计不做新数值运行。 |
| F02 姿态log的小角近似/域裁剪 | p2,3 / 印刷3994,3995，1：只列三个旋转分量，未规定SO(3)对数实现。 | acos前把trace量裁剪[-1,1]；theta<1e-6用0.5*vee；其余theta/sin(theta)，没有独立的接近pi稳定分支。 [kinematics.py:9–14](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/kinematics.py:9)；[kinematics.py:137–145](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/nmpcc/kinematics.py:137) | **数值保护与奇异域限制**。小角近似、裁剪合理且应披露；接近pi存在表示/导数风险。当前欧氏rotvec差也不是完整相对旋转对数。 限制：未证明当前手持姿态到达危险域，不能描述为已发生故障。 |
| F03 启动IK与物理接触数值选项 | p8,9 / 印刷4000,4001，§V setup; Experiment II：真实Panda与实体箱体，未给MuJoCo IK、软接触或noslip。 | IK阻尼伪逆、关节clamp只用于初始化；手持屏蔽非pad机器人碰撞；物理软接触/摩擦/noslip与预测理想弹簧是两套模型。 [handheld.py:78–94](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scenes/handheld.py:78)；[handheld.py:129–145](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scenes/handheld.py:129)；[ik.py:174–188](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/ik.py:174)；[scene.py:58–62](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scene.py:58) | **初始化/plant设定，非论文控制律**。新ideal未把真实MuJoCo接触改成理想材料。禁碰撞会缩小仿真安全问题范围，不能用此结果证明真实双臂无碰撞。 限制：未全面审计所有XML资产及底层碰撞算法；当前接触参数以保存MJB为准。 |
| F04 滤波与日志时刻 | p6,9 / 印刷3998,4001，§III-D; Fig10/12：原文没有足够信息确认所有传感滤波/时延。 | measure_contact直接累加，无移动平均/低通；日志每5步。flat legacy_csv_t偏移+.002，handheld用步前t；非控制日志行可能携带上一控制拍预测。 [measurements.py:18–60](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/diagnostics/measurements.py:18)；[recording.py:132–142](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/recording.py:132)；[recording.py:156–177](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/recording.py:156) | **指定调用链未见测量低通；有降采样**。不要把降采样曲线或同标签时刻视为同一测量时刻；物理/控制/预测须分别对齐。参考smoothstep不是力测量滤波。 限制：结论限于列明调用链及源码扫描；不声称外部库、驱动或论文设备完全无滤波。 |
| F05 实验对象与性能指标 | p8,9,10,11 / 印刷4000,4001,4002,4003，§V; Fig12; Conclusion：双Panda、箱长1.2m/0.9kg和4空罐；实验II报告y同步误差<2mm、运动阶段力同步误差<2N。 | OpenArmX；flat箱长.5m/质量.8kg，手持BOX_HALF=[.12,.20,.10]。当前比较还使用各侧对10N的RMS、终点位置误差、触限等。 [exp2_slender_scene.py:56–67](D:/RobotSim/results/ideal_contact_handheld_20261012/paper_source_snapshot/exp2_slender_scene.py:56)；[handheld.py:8–14](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/scenes/handheld.py:8)；[handheld_output.py:17–27](D:/RobotSim/results/ideal_contact_handheld_20261012/candidate_cop08_f02_move05/source/experiments/legacy/handheld_output.py:17) | **移植实验，不是原场景逐值复现**。左右力同步小不意味着每侧接近10N；稳态13/13N可同步良好仍对参考偏3N。新版评估应同时保留两者。 限制：论文图12(d)原始数值/作者代码未获取，不能将本项目各侧RMS当论文同一指标。 |

## 核心公式与判读边界

当前实现的每臂组合误差为

$$
E_i=(I-\eta_{f,i})[r_i-(I-\eta_{cop,i})r_{i,r}-\eta_{cop,i}r_j]-\eta_{f,i}k_f(F_i-F_{i,r}).
$$

这里 \(r_i\) 是tip世界位置及旋转向量，\(r_{i,r}\) 为参考，\(F_i\) 为bias校正后的预测六维反馈；\(\eta_{cop,i}\)、\(\eta_{f,i}\) 是逐轴混合系数，\(k_f\) 是本项目额外的力误差缩放。论文式12对前面的协作项与力项允许在同一残差内抵消；因此此前保持段发现的位置项与力项抵消本身不是反号证据。

论文式12又定义 \(\varepsilon_i=E_i^T E_i\)。若将它原样代入式31/32，那么该项的二次代价为 \((E_i^TE_i)^2\)，其导数为 \(2E_i^T\dot E_i\)。当前代码采用 \(E_i\)、\(\dot E_i\) 的二次代价。这里必须保留两种可能：实现有意采用常见的向量残差LS，或论文符号在后式中复用；本次没有官方源码证据可以替作者决定。原文直接相加位置与力也需要隐含单位归一化或有量纲权重，但论文没有给出足够单位约定；不能因量纲疑问就认定作者实际上采用当前kf。该差异比只调整kx更根本，不能悄悄改变。

新手持模型在一次控制周期内使用

$$
c=g_0-(x_{R,face}-x_{L,face}),\qquad F_{nom,L}=-k_x\max(c,0),\qquad F_{nom,R}=-F_{nom,L},
$$

$$
b_i=F_{meas,i}-F_{nom,i}(q_{now}),\qquad F_{pred,i}(q)=F_{nom,i}(q)+b_i.
$$

\(g_0\) 是接触面静止间距，\(c\) 是两面中心世界x间距得到的压缩代理量，\(k_x\) 是预测系数，\(b_i\) 在预测期内冻结。实现选择 \(c>0\) 时正部导数为1、\(c\le0\) 为0。**这只保证名义弹簧单向，不能保证总预测力在名义未压缩区为零。** 此外，c<=0不等于真实物理接触消失：倾斜面仍可能有边角接触，必须看实际接触几何。 冻结残差同样可能保留当前切向力与力偶；不得据此声称具有真实接触开闭预测。

每周期预测加速度为

$$
\ddot q_i=M_{i,now}^{-1}[u_i+\tau_{res,i}-G_{i,now}-D_i\dot q_i+J_{face,i}(q_i)^T(F_{pred,i,x},0,0)^T].
$$

\(M_{i,now}\)、\(G_{i,now}\)、\(D_i\)、\(\tau_{res,i}\) 是周期采样后冻结的参数；\(J_{face,i}\) 随预测关节位置更新。论文式25/35的状态相关惯性/重力与速度耦合项没有被完整复现。该模型仍因FK和接触几何而非线性，不能把“非线性”误解成“完整刚体动力学”。

## 手持与平放的实际差异

以下为已测试候选与平放的差异；最终生产手持默认也为 smooth_positive，可用显式 CLI 选择 ideal_unilateral。

| 项目 | 已测试手持理想候选 | 平放对照 |
|---|---|---|
| 名义接触律 | ideal_unilateral，kx=2419.3 | smooth_positive，kx=2793.166 |
| 控制法向反馈 | signed world Fx | 左负右正的normal幅值和 |
| bias范围 | legacy六轴；力矩不含参考点力臂 | 仅x |
| 初始化 | 二分实测预压校准press0 | 几何刚度映射预压 |
| 控制左右 | 对物理左右交换命名 | 沿场景原约定 |
| rest_gap | 从几何值减pad两半厚到face口径 | 使用flat标定常量 |
| 共同项 | 冻结M/G/接触残差、缺显式Coriolis、kf=.003、相同误差/代价与6ms控制周期 | 同左 |

上述行均由C01/C04/C05/C06/D02的代码锚点支持。两场景共享核心不等于所有物理假设相同；flat保留平滑是此次授权范围，未自动替换。

## 可确认与仍未知

论文完整论证包括：四种合作模式的定义与静态关系；两连杆数值例（明确忽略摩擦）；双Panda的同步、夹持运输、人机交互三种实验；以及用人类动作示范模式切换的附录。它支持这套混合任务指标在所述实验中的作用，未提供当前OpenArmX/MuJoCo程序的稳定性、安全性或接触参数证明。论文p6明确略去稳定性分析，而非对本实现给出证书。

本次检查了论文链接、IEEE文章页、华中科技大学作者张宇豪页面及论文标题/DOI的代码检索。IEEE页面受验证页限制，作者页列出论文但未见对应代码链接；**在这些已检查页面未找到可钉住版本的作者官方实现**，不是断言作者没有源码。当前仓库是用户项目，不能把其历史archive的oracle当论文官方源码。

尚未知：作者式31/32的epsilon真正实现；原离散终端输入定义；原N、积分器、正则/容差、具体Q/P/导数/输入权重；kf类隐式归一化；Panda力估计滤波/延迟；实际每预测节点M/C/G更新策略；本文2419.3历史标定的完整原始数据来源。这些项目均不得以猜测补齐。

## 后续检查顺序（建议，不执行）

1. 先保留当前代码/旧结果与本审计证据，分别标注“论文概念移植”和“逐公式一致性”，避免将新ideal称作论文原型。
2. 先澄清A03的E/epsilon与E02的终端u，获得作者代码或明确采用哪种目标；这些会改变优化问题，不应与接触律变更混在一次对照中。
3. 若继续查偏差，对同一保存状态逐项比较真实Coriolis、冻结M/G误差、未来bias/接触开闭；单独统计当前测量与未来预测，不能用锚定节点的相等掩盖预测误差。
4. 对新ideal闭环仍验收每2ms实际限位/接触/相对滑移、终态和未来节点，而非只看solver status。改变轨迹或参数的结果分别列明，不能混表掩盖旧失效。

本轮未执行项目代码、仿真、求解器或修改核心；只读取源文件/PDF、复用旧证据、提取文本并生成审计文档。报告结论不授权额外调参。
