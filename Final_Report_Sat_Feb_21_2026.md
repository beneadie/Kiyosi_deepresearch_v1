# 全球具身智能（Embodied AI）技术路线与产业格局深度调研报告（2026年版）

## 执行摘要

截至2026年2月，全球具身智能领域已正式跨越“技术验证期”，进入“规模化量产与工业深度部署”的元年。2025年见证了视觉-语言-动作模型（VLA）和世界模型（World Models）的突破性进展，使得机器人能够处理非结构化环境下的复杂任务。在产业格局上，形成了以北美（算法与生态主导）和中国（供应链与快速量产主导）为核心的双极态势。2025年全球人形机器人出货量突破1.8万台，其中中国厂商占据了显著的市场份额。本报告将深入探讨当前主流技术路线、识别全球代表性企业，并详细分析其技术路径、商业化进展及融资背景。

---

## 主流技术路线演进：从分层控制到端到端智能

当前具身智能的技术演进呈现出从传统的“分层控制”向“端到端大模型”演进的趋势，核心目标是实现感知、决策与控制的高度集成。这种转变的核心在于如何让机器人像人类一样，通过观察和模仿来学习复杂的物理交互。

### 视觉-语言-动作模型（VLA）的崛起
端到端架构主张将传感器输入直接映射为执行器的动作指令，减少了人工设计的中间环节。基于Transformer架构的视觉-语言-动作模型（VLA）已成为当前最前沿的技术标准。这些模型通过大规模多模态数据训练，使机器人具备了跨硬件形态的泛化能力。例如，Google DeepMind在2025年底发布的VLA 2.0（Gemini Robotics On-Device）实现了在机器人本地硬件上的低延迟运行，其显著特征是能够从仅50至100次的人类演示中学习新技能，通用性得分较传统模型提升了一倍以上 [15, 17, 19]。这种模型不仅能理解文字指令，还能通过视觉反馈实时调整动作，解决了机器人在非结构化环境中“死板”的问题。

### 世界模型（World Models）与物理直觉
世界模型赋予机器人预测物理世界变化的能力，使其具备某种程度的“物理直觉”。通过预测动作发生后的视觉反馈，机器人在虚拟仿真中进行数亿次“想象”训练。Google的Genie 3和NVIDIA的Cosmos平台是这一路线的代表。Genie 3能从被动视频中生成物理一致的3D交互世界，作为机器人训练的“现实模拟器”，极大地降低了在物理世界中收集数据的成本和风险 [10, 19]。Wayve发布的GAIA-2模型则通过2500万条视频序列训练，实现了对复杂交通场景的高保真模拟，支持生成罕见和高风险场景（如突然切入、紧急避让）以供机器人策略验证 [20, 22]。

### 强化学习与模拟到现实（Sim-to-Real）的闭环
利用大规模并行强化学习在物理仿真引擎（如NVIDIA Isaac）中进行试错，依然是提升机器人运动控制稳定性的关键。2025年的技术突破在于将触觉反馈集成到VLA模型中，使得机器人在接触密集型任务（如灵巧操作、精密组装）中的成功率提升了15%以上 [8]。此外，分层控制架构在处理长程任务规划（Long-horizon planning）时仍具有优势，即利用大语言模型（LLM）作为“大脑”进行任务拆解，底层采用模型预测控制（MPC）或全身控制（WBC）算法，这种架构在处理需要常识推理的复杂任务时具有更强的可解释性和安全性 [18]。

---

## 全球产业格局：北美算法高地与中国量产先锋

具身智能的产业竞争已演变为一场关于“数据、算力与供应链”的综合竞赛。北美阵营凭借顶尖的AI人才储备和自研算力芯片，在通用AI大脑的研发上保持领先；而中国阵营则依托强大的制造业基础，在硬件迭代速度和成本控制上展现出极强的竞争力。

### 北美阵营：Scaling Law在硬件上的复现
以Tesla、Figure AI和Physical Intelligence为代表的北美企业，更倾向于追求“通用性”和“Scaling Law”在硬件上的复现。其优势在于极高的估值和资本密集度。2025年，Figure AI的估值已达到390亿美元，而Physical Intelligence在完成6亿美元B轮融资后，估值也攀升至56亿美元 [5, 40]。这些企业致力于构建硬件无关的“通用机器人大脑”，试图定义具身智能的底层操作系统。

### 中国阵营：供应链优势与“中国速度”
以智元机器人（Agibot）、宇树科技（Unitree）为代表的中国企业，核心竞争力在于极高的硬件迭代速度。2025年，中国厂商在人形机器人出货量上占据全球领先地位。智元机器人实现了5000台级别的量产交付，而宇树科技的年度出货量也超过了5500台 [1, 26]。中国企业通过建立“数据工厂”和推出机器人租赁平台，快速积累真实世界的交互数据，形成了“量产带动数据，数据优化算法”的闭环。

---

## 代表性公司深度分析：北美巨头与初创先锋

### Tesla (Optimus)
Tesla的技术路径深度集成其FSD（全自动驾驶）的视觉神经网络，采用完全端到端的控制方案。2026年初发布的Optimus Gen 3在硬件上取得了飞跃，其单手拥有25个执行器（双手共50个），精度较前代提升一倍，能够执行极其精细的组装任务 [7, 8]。在商业化方面，Tesla已在德州超级工厂部署数百台Optimus执行电池组装任务，并计划在2026年底前将Fremont工厂的部分生产线转为人形机器人专用线，目标年产100万台 [9]。其核心团队由Elon Musk直接领导，成员主要来自FSD算法团队和Autopilot硬件团队，这种“车机协同”的模式使其在视觉感知和大规模制造上具有天然优势。

### Figure AI
Figure AI坚持与OpenAI深度合作，利用多模态大模型赋予机器人语音交互与逻辑理解能力。其Figure 03模型在2025年底被誉为实现了“物理劳动的GPT-3时刻”。在宝马（BMW）Spartanburg工厂的部署中，Figure 03表现出400%的效率提升，单台机器每天工作10小时，连续运行5个月装载超过9万个零件而无重大机械故障 [6]。Figure AI在2025年完成了C轮融资，估值达390亿美元，并建立了名为“BotQ”的设施，旨在实现年产1.2万台人形机器人的目标 [5, 6]。

### 1X Technologies
1X Technologies采用专利的绳驱（Tendon Drive）系统，强调安全的人机交互，这使其在家庭服务场景中具有独特优势。2026年2月，公司发布了家用机器人NEO Gamma，并开启了在“数百至数千个”家庭中的Beta测试 [34, 35]。尽管测试显示机器人在执行装载洗碗机等任务时仍显缓慢（需5分钟），且部分任务需远程操作员（Turing Pilots）辅助，但其2万美元的售价和每月499美元的订阅模式已引起市场高度关注 [36, 38]。1X在2025年寻求10亿美元融资，目标估值100亿美元，核心支持者包括OpenAI和Tiger Global [34, 35]。

### Physical Intelligence (π)
作为一家专注于“机器人通用大脑”的初创公司，Physical Intelligence在2025年发布了π0系列模型。其π0.6模型基于Gemma3 4B构建，能够直接执行折叠衣服、清理卧室等任务，且在无需特定任务微调的情况下，盒子组装的成功率达到了20% [43, 44]。公司在2025年完成6亿美元B轮融资，由CapitalG领投，亚马逊创始人贝佐斯、OpenAI等参投 [41, 43]。其核心团队由来自Google DeepMind、UC Berkeley的顶尖科学家组成，致力于开发硬件无关的AI框架。

---

## 代表性公司深度分析：中国量产领跑者

### 智元机器人 (Agibot)
智元机器人是全球首个实现人形机器人大规模量产的企业，2025年12月完成第5000台机器人下线，占据全球约39%的市场份额 [1, 2, 5]。其技术路径基于自研的“远征”系列架构，搭载WorkGPT大模型，多模态输入处理准确率达96% [6]。智元在商业模式上极具创新，于2025年底推出了“青天租”机器人租赁平台，覆盖50个城市，A2机器人日租金约9800元人民币，旨在通过租赁模式降低工业客户的试用门槛 [7, 8]。公司在2025年8月获LG电子、未来资产战略投资，估值超150亿人民币，创始人为知名技术博主彭志辉（稚晖君） [1, 2]。

### 宇树科技 (Unitree)
宇树科技凭借在四足机器人领域积累的运动控制经验，迅速在人形机器人市场取得领先。2025年其人形机器人出货量超过5500台，超越了多家美国竞争对手的总和 [26, 27]。2026年初，宇树发布了工业级H2机器人，具备更强的负载能力和自主集群控制功能，能够在无需外部跟踪系统的情况下实现数十台机器人的协同运动 [11, 12]。其G1折叠人形机器人售价仅为1.6万美元，极大地推动了科研教育市场的普及 [11, 13]。宇树目前正计划在上海科创板IPO，目标估值70亿美元 [11]。

### 银河通用 (Galbot) 与 逐际动力 (LimX Dynamics)
银河通用在2025年6月获得11亿人民币融资，并与博世（Bosch）旗下的博原资本成立合资公司，专注工业制造应用。其G1半人形机器人（轮式底盘+双臂）已在北京多家零售店部署，执行补货和导航任务 [49, 54, 55]。逐际动力则在2026年2月完成2亿美元B轮融资，投资方包括上汽、蔚来、京东等。其全尺寸人形机器人CL-1已开启预售，售价约15.8万人民币，核心技术在于其模块化硬件平台TRON 2和认知操作系统COSA [57, 59, 60]。

---

## 产业影响因素分析：人才流动、并购与市场预测

### 人才大迁移与“AI人才战争”
2025年见证了从顶级AI实验室向机器人初创企业的“人才出埃及记”。新成立的Periodic Labs吸引了超过20名来自OpenAI、Google DeepMind和Meta的顶尖研究员，其中包括ChatGPT的共同创造者Liam Fedus [11, 56]。Meta也组建了超智能实验室（MSL），通过高达1亿美元的签字费从OpenAI等机构挖角，由Alexandr Wang和Nat Friedman领导 [57]。这种人才的跨界流动加速了前沿AI算法在物理硬件上的落地。

### 并购动态与战略整合
并购活动在2025年达到高峰，反映了行业向垂直整合发展的趋势。软银集团以53.75亿美元收购了ABB的机器人部门，旨在加强其在自动化领域的布局 [16, 66]。Meta则以超过20亿美元收购了AI智能体初创公司Manus [18, 68]。此外，波士顿动力与丰田研究院（TRI）及Google DeepMind达成战略合作，将大模型集成至Atlas机器人，标志着硬件巨头与算法巨头的深度结盟 [6, 19]。

### 市场规模与未来展望
根据IDC的数据，2025年全球人形机器人市场收入达4.4亿美元，同比增长508%，预计到2030年，具身智能机器人将占据机器人总市场的30%以上 [9, 10]。Fortune Business Insights预测，全球人形机器人市场规模将从2025年的48.9亿美元增长至2034年的1651.3亿美元，复合年增长率达50.6% [63]。然而，Gartner也发出预警，认为到2028年，全球预计只有不到20家公司能真正实现AI人形机器人的规模化生产，技术成熟度和高昂的成本仍是主要障碍 [64]。

---

## 具身智能产业核心要素映射表

| 核心维度 | 关键因素 | 具体影响与案例 |
| :--- | :--- | :--- |
| **技术路径** | VLA模型与世界模型 | 提升非结构化环境泛化能力；如Gemini Robotics VLA 2.0 [15, 19] |
| **量产能力** | 供应链集成与数据工厂 | 降低成本并加速算法迭代；如智元机器人年产5000台 [1, 5] |
| **商业模式** | 租赁平台与订阅制 | 降低客户门槛，加速场景渗透；如1X NEO订阅模式 [36, 37] |
| **人才流动** | 顶级实验室人才外流 | 加速AGI技术向物理世界迁移；如Periodic Labs的成立 [11, 56] |
| **资本投入** | 巨额融资与高估值 | 支持长周期研发与设施建设；如Figure AI 390亿美元估值 [5, 6] |

---

### Sources

[1] AgiBot First to Ship 5,000 Humanoid Robots - Chosun English: https://www.chosun.com/english/world-en/2025/12/26/X6UYTLNXIFFV7I5WPNQIVLMHBY/
[2] AGIBOT tops the global humanoid robot shipments ranking - Gasgoo: https://autonews.gasgoo.com/articles/news/agibot-tops-the-global-humanoid-robot-shipments-ranking-2009638041914548225
[3] Two-thousand humanoid robots deployed for Spring Festival - Embassy of China: https://www.facebook.com/100090874324645/posts/two-thousand-humanoid-robots-are-being-deployed-across-200-cities-in-china-to-he/861595933546210/
[4] Figure 02 Review: Price, Specs & BMW Deployment Data [2026]: https://blog.robozaps.com/b/figure-02-review-2026
[5] Figure AI - Wikipedia: https://en.wikipedia.org/wiki/Figure_AI
[6] Figure AI Achieves 400% Efficiency Gain at BMW's Spartanburg Plant - Chronicle Journal: https://markets.chroniclejournal.com/chroniclejournal/article/tokenring-2026-1-21-the-humanoid-inflection-point-figure-ai-achieves-400-efficiency-gain-at-bmws-spartanburg-plant
[7] Tesla Optimus Gen 3 Hands Revealed: 50-Actuator Precision Leap - BASENOR: https://www.basenor.com/blogs/news/tesla-optimus-gen-3-hands-revealed-50-actuator-precision-leap
[8] Tesla's Robotic Moonshot: Optimus Gen 3 - Not A Tesla App: https://www.notateslaapp.com/news/3281/teslas-robotic-moonshot-optimus-gen-3
[9] Tesla's Third-Generation Robot to Debut, Targeting 1 Million Units - Gasgoo: https://autonews.gasgoo.com/articles/news/teslas-third-generation-robot-to-debut-targeting-1-million-units-in-annual-production-this-year-2018729049449140225
[10] The House of Google: Beyond Gemini - Saanya Ojha Substack: https://saanyaojha.substack.com/p/the-house-of-google-beyond-gemini
[11] Top A.I. Researchers Leave OpenAI, Google and Meta for Periodic Labs - NYT: https://www.nytimes.com/2025/09/30/technology/ai-meta-google-openai-periodic.html
[12] Interleave-VLA: Enhancing Robot Manipulation with Image-Text - OpenReview: https://openreview.net/forum?id=ULTWUuGhC3
[13] Vision Language Action (VLA) models breakthrough - Exxact Corp: https://www.facebook.com/exxactcorp/posts/vision-language-action-vla-models-are-a-breakthrough-for-robotics-they-combine-v/1303321385152465/
[14] 24 AI startups to watch in 2026 - LinkedIn (Paulo Ellis): https://www.linkedin.com/posts/pauloellis_24-ai-startups-to-watch-in-2026-activity-7389846281327882244-EyKi
[15] Gemini Robotics On-Device brings AI to local robotic devices: https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/
[16] ABB divests Robotics division to SoftBank Group: https://www.robotics247.com/article/top_10_robotics_automation_investment_funding_mergers_and_acquisitions_news_of_2025
[17] Google DeepMind Announces Robotics Foundation Model Gemini: https://www.infoq.com/news/2025/07/google-gemini-robotics/
[18] Meta acquired the AI agent startup Manus for over $2 billion: https://aidatainsider.com/ai/2025s-top-16-acquisitions-in-ai-data/
[19] Gemini Robotics: Next-Generation Robot AI by Google: https://zenn.dev/katsuhisa_/articles/gemini-robotics-overview
[20] Wayve: GAIA-2 Technical Report (March 2025): https://wayve.ai/wp-content/uploads/2025/03/GAIA_2_Technical_Report.pdf
[21] Fenxi: Wayve GAIA-1 Generative World Model Impact: https://fenxi.fr/en/blog/gaia-1-wayve-generative-world-model-impact/
[22] Wayve: Generative AI for video generation and simulation: https://wayve.ai/science/gaia/
[23] SONIC-O1: A Real-World Benchmark for Evaluating Multimodal (2026): https://vectorinstitute.github.io/sonic-o1/
[24] Scouts by Yutori: VLA model updates (Dec 2025): https://scouts.yutori.com/35b756a0-e717-4849-85cd-d22f5ce21709
[25] Google DeepMind Models: https://deepmind.google/models/
[26] SCMP: China's Unitree ships more than 5500 humanoid robots in 2025: https://www.scmp.com/tech/tech-trends/article/3340446/chinas-unitree-ships-more-5500-humanoid-robots-2025-surpassing-us-peers
[27] Tech in Asia: China's Unitree ships over 5500 humanoid robots in 2025: https://www.techinasia.com/news/chinas-unitree-ships-over-5500-humanoid-robots-in-2025
[28] XMAQUINA: 2025's Hottest Humanoid Robots: https://www.xmaquina.io/blog/2025s-humanoid-robots-how-do-they-stack-up
[29] Reddit: Top 10 AI Updates Today (Feb 17, 2026): https://www.reddit.com/r/AIPulseDaily/comments/1r7dwat/top_10_ai_updates_today_feb_17_2026_the_week_that/
[30] LM Council: AI Model Benchmarks Feb 2026: https://lmcouncil.ai/benchmarks
[31] VisuLogic Benchmark Review (April 21, 2025): https://liner.com/review/visulogic-benchmark-for-evaluating-visual-reasoning-in-multimodal-large-language
[32] VisuLogic Project Page: https://visulogic-benchmark.github.io/VisuLogic/
[33] China's AgiBot Launches Robot Leasing Platform With Daily Rates: https://www.yicaiglobal.com/news/chinas-agibot-launches-robot-leasing-platform-with-daily-rates-up-to-usd14227
[34] Backed by OpenAI, 1X Technologies Aims for Up to USD 1 Billion: https://equalocean.com/briefing/20250924230148618
[35] In Less Than a Year, This OpenAI-approved Company Has Grown 12x: https://www.ainvest.com/news/year-openai-approved-company-grown-12x-valuation-2509/
[36] OpenAI-backed startup aims to deliver in-home humanoid robots: https://sifted.eu/articles/1x-humanoid-robot-launch
[37] NEO Home Robot | Order Today: https://www.1x.tech/discover/neo-home-robot
[38] 1X's $20000 NEO Humanoid Robot Launches With A Catch: https://dronexl.co/2025/10/29/1xs-neo-humanoid-robot-launches-chores/
[39] 1X NEO Review: $20K Home Robot [2026]: https://blog.robozaps.com/b/1x-neo-release-date-rumors-news
[40] Bloomberg: Robotics Startup Physical Intelligence Valued at $5.6 Billion: https://www.bloomberg.com/news/articles/2025-11-20/robotics-startup-physical-intelligence-valued-at-5-6-billion-in-new-funding
[41] Physical Intelligence (π) Official Website: https://www.pi.website/
[42] MLQ.ai: Physical Intelligence Startup Nears $1 Billion Funding Milestone: https://mlq.ai/news/physical-intelligence-startup-nears-1-billion-funding-milestone-for-robot-ai-development/
[43] Marktechpost: Researchers at Physical Intelligence Introduce π-0.5: https://www.marktechpost.com/2025/04/22/researchers-at-physical-intelligence-introduce-%CF%80-0-5-a-new-ai-framework-for-real-time-adaptive-intelligence-in-physical-systems/
[44] π0.6 Model Card (Nov 17, 2025): https://website.pi-asset.com/pi06star/PI06_model_card.pdf
[45] LearnOpenCV: Vision Language Action Models (VLA) & Policies for Robots: https://learnopencv.com/vision-language-action-models-lerobot-policy/
[46] CRN: 10 AI Startup Companies To Watch In 2025: https://www.crn.com/news/ai/2024/10-ai-startup-companies-to-watch-in-2025
[47] PHR Robotics: The Best Intelligent Robots in 2025: https://www.phr-robotics.com/en/blog/meilleurs-robots-intelligents-en-2025
[48] Salesforce Ventures: The Robotics Breakout Moment: https://salesforceventures.com/perspectives/the-robotics-breakout-moment/
[49] Galbot picks up $153M to commercialize G1 semi-humanoid: https://www.therobotreport.com/galbot-picks-up-153m-commercialize-g1-semi-humanoid/
[50] China's Unitree Aims to Ship 20,000 Humanoid Robots in 2026: https://www.eweek.com/news/unitree-20000-humanoid-robots-2026-china/
[51] China's Kung Fu Humanoid Robots Rise - YouTube: https://www.youtube.com/watch?v=9x4fK7R7VAE
[52] Unitree targets 20,000 humanoid robots with fourfold capacity increase: https://interestingengineering.com/ai-robotics/unitree-targets-20000-humanoid-robots
[53] China's Unitree aims to ship 20,000 humanoid robots in 2026: https://www.techinasia.com/news/chinas-unitree-aims-to-ship-20000-humanoid-robots-in-2026
[54] China's Galbot Secures $153M, Launches Humanoid Robotics JV: https://asiatechdaily.com/chinas-galbot-secures-153m-launches-humanoid-robotics-jv-with-boyuan-capital/
[55] Galbot Raises $153 Million to Expand Embodied AI Robots: https://theaiinsider.tech/2025/06/25/galbot-raises-153-million-to-expand-embodied-ai-robots-partners-with-bosch-group-investment-arm/
[56] Boyuan Capital and Galbot Launch Joint Venture: https://finance.yahoo.com/news/boyuan-capital-investment-platform-under-055300861.html
[57] China's LimX Dynamics Raises $200M in Series B Funding: https://theaiinsider.tech/2026/02/02/chinas-limx-dynamics-raises-200m-in-series-b-funding-for-real-world-embodied-robots/
[58] The companies making China a robotics powerhouse: https://www.techinasia.com/visual-story/companies-making-china-robotics-powerhouse
[59] Chinese humanoid robot maker LimX secures $200 million: https://cnevpost.com/2026/02/02/chinese-humanoid-robot-maker-limx-secures-200-million-in-series-b-funding/
[60] LimX raises $200M to build embodied intelligence: https://siliconangle.com/2026/02/02/limx-raises-200m-build-embodied-intelligence-humanoid-robotics/
[61] Grand View Research: Humanoid Robot Market Size & Share 2030: https://www.grandviewresearch.com/industry-analysis/humanoid-robot-market-report
[62] MarketsandMarkets: Embodied AI Market Reports: https://www.marketsandmarkets.com/Market-Reports/embodied-ai-market-28244215.html
[63] Fortune Business Insights: Humanoid Robot Market Size, Share, & Growth Report [2034]: https://www.fortunebusinessinsights.com/humanoid-robots-market-110188
[64] Gartner Predicts Fewer Than 20 Companies Will Scale Humanoid Robots: https://www.gartner.com/en/newsroom/press-releases/2026-01-21-gartner-predicts-fewer-than-20-companies-will-scale-humanoid-robots-for-manufacturing-and-supply-chain-to-production-stage-by-2028
[65] Agility Robotics valuation, funding & news | Sacra: https://sacra.com/c/agility-robotics/
[66] Top 10 robotics & automation investment, funding, mergers and acquisitions news of 2025: https://www.robotics247.com/article/top_10_robotics_automation_investment_funding_mergers_and_acquisitions_news_of_2025
[67] Sanctuary Cognitive Systems Closes C$75.5 Million Series A: https://www.sanctuary.ai/blog/sanctuary-ai-closes-75-million-series-a-funding
[68] These 9 big deals defined the year in artificial intelligence: https://www.businessinsider.com/biggest-ai-deals-acquisitions-of-the-year-2025-12