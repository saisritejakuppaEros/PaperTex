001 Supplementary Material: 001
002 RefCompose: Multi-Reference Image Generation 002
003 via LoRA-Conditioned Diffusion 003
004 Anonymous ECCV 2026 Submission 004
005 Paper ID #8 005
006 A Overview 006
007 This supplementary document accompanies the main ECCV workshop paper on 007
008 RefCompose. The main paper focuses on Dense Layout quantitative evaluation 008
009 on InstanceAssemble [16]; here we provide: 009
010 1. Layout-centric benchmarks (Section B) comparing the prompt-induced 010
011 spatial layout stage against dedicated layout generators on COCO-GR [6] 011
012 numerical and spatial reasoning splits. 012
013 2. Extended ablation study (Section C) with quantitative metrics comple- 013
014 menting the qualitative analysis in the main paper. 014
015 3. Training data details (Section D), including the synthetic dataset prepa- 015
016 ration pipeline, referenced from the main experiments section. 016
017 Throughout, multiple visual references are composed under a single scene prompt 017
018 without annotated bounding boxes or per-reference token scaling. 018
019 B Layout-Centric Evaluation 019
020 We situate the spatial layout stage of RefCompose in the broader layout-generation 020
021 literature by comparing against LayoutGPT [6] and Lay-Your-Scene [13] on 021
022 COCO-GR [6] numerical and spatial reasoning splits. RefCompose first induces 022
023 a coarse composition with a frozen text-to-image model (Flux 2.0 Dev [3]) and 023
024 then extracts object boxes with Grounding DINO [11]. 024
025 Metrics. L-FID evaluates layout realism by rendering predicted layouts as 025
026 colour-coded object maps and computing an FID-style distance to ground-truth 026
027 layouts [7]; lower is better. For numerical reasoning, precision, recall, and accu- 027
028 racy measure whether the predicted object set matches the prompt-specified ob- 028
029 jects. Numerical GLIP-Accuracy checks whether the requested object counts 029
030 appear in the generated image [9]. For spatial reasoning, accuracy measures 030
031 whether predicted boxes satisfy relations such as left, right, above, or below; 031
032 spatial GLIP-Accuracy checks whether those relations remain visible after 032
033 image generation [9]. 033
2 ECCV 2026 Submission #8
Table S1: Layout quality on COCO-GR [6] (lower L-FID is better).
Method L-FID↓
LayoutGPT [6] 3.51
Lay-Your-Scene [13] 3.07
RefCompose (spatial layout) 2.98
Table S2: Numerical and spatial reasoning on COCO-GR [6]. GLIP-Accuracy measures image-level count/spatial compliance [9].
Numerical Reasoning Spatial Reasoning
Method Prec.↑ Recall↑ Acc.↑ GLIP↑ Acc.↑ GLIP↑
LayoutGPT [6] 76.29 86.64 76.72 54.25 87.07 56.89
Lay-Your-Scene [13] 77.62 99.23 95.14 56.20 92.58 58.94
RefCompose (spatial layout) 78.21 99.25 96.41 56.29 93.13 59.39
034 Results. Table S1 reports layout quality; Table S2 reports numerical and spa- 034
035 tial reasoning. RefCompose attains the best L-FID and spatial accuracy among 035
036 compared methods. We attribute this to inverse layout construction: a large pre- 036
037 trained text-to-image model [3] first produces an aesthetically composed scene, 037
038 after which boxes are recovered from visible object support. Unlike LLM plan- 038
039 ners that reason symbolically [6], or layout diffusion transformers trained on 039
040 narrower layout corpora [13], the T2I prior encodes broad composition statistics 040
041 from large-scale pretraining; sharper extracted boxes then improve downstream 041
042 canvas placement and image-level spatial compliance. 042
043 C Extended Ablation Study 043
044 The main paper presents a qualitative ablation of the reference canvas and Depth 044
045 Anything 3 [10] depth prior (Figure 7 in the main document). Table S3 reports 045
046 the corresponding quantitative results on the Dense Layout test split [16], us- 046
047 ing the same evaluation protocol as Table 2 in the main paper (DINO [5] for 047
048 composition and reference consistency). 048
049 Discussion. Removing the reference canvas causes the largest drop on Color, 049
050 Texture, DINOrefs, and CLIP-I (appearance-centric signals), while Spatial re- 050
051 mains relatively high because the depth prior still constrains placement. The 051
052 model then infers appearance from text alone, landing near the CreatiLay- 052
053 out/InstanceAssemble [14, 16] range on photometric metrics. CLIP-T increases 053
054 slightly, mirroring the text-alignment trade-off discussed in the main paper. 054
055 Removing the depth prior causes the largest drop on Spatial and Shape, re- 055
056 flecting degraded placement and occlusion reasoning, while Color, Texture, and 056
057 DINOrefs remain closer to the full model since the canvas still supplies appear- 057
058 ance. This matches the qualitative observation in the main paper that subjects 058
ECCV 2026 Submission #8 3
Table S3: Ablation study evaluating the contributions of the reference canvas and
depth prior. The full model consistently achieves the best performance across identity
preservation, spatial arrangement, and semantic alignment metrics.
Configuration Spatial↑ Color↑ Texture↑ Shape↑ PICK↑ DINO↑ DINOrefs↑ CLIP-I↑ CLIP-T↑
Full model (canvas + depth, dual LoRA) 0.908 0.826 0.854 0.849 0.229 0.959 0.613 0.766 0.327
w/o reference canvas (text-only appearance) 0.861 0.542 0.571 0.558 0.224 0.712 0.471 0.702 0.334
w/o depth prior (canvas only, no structure) 0.683 0.781 0.809 0.612 0.221 0.798 0.587 0.741 0.310
059 such as the bat and bed remain recognizable, but their arrangement becomes 059
060 contextually implausible. Together, these results confirm that canvas and depth 060
061 provide complementary, disentangled signals for appearance and structure. 061
062 D Training Data Details 062
063 As noted in the main paper, RefCompose is trained on a mixture of a private 063
064 movie dataset and synthetic composites. 064
065 Real data. We curate ultra high-resolution movie frames at 1280 × 720 from 065
066 a private dataset sourced from movies. Object references are canonicalized to 066
067 neutral pose and lighting to reduce spurious correlations between scene context 067
068 and subject appearance. The five-stage pipeline described in the main paper 068
069 (Qwen3-VL [1] enumeration, Grounding DINO [11] localisation, Flux 2.0 Dev [3] 069
070 canonicalization, DINO [5] filtering, and re-annotation) yields approximately 070
071 50,000 samples from 1,000 distinct films. 071
072 Synthetic data. We complement real cinematic frames with synthetically gener- 072
073 ated composites that expose the model to diverse spatial arrangements, lighting 073
074 conditions, object interactions, and viewpoint configurations underrepresented in 074
075 canonicalized movie crops. The preparation pipeline is described below; together 075
076 with real data, this mixture encourages generalisation beyond studio lighting and 076
077 fixed poses without sacrificing photometric fidelity on in-domain cinematic con- 077
078 tent. 078
079 Synthetic dataset preparation pipeline. Our synthetic data pipeline consists of 079
080 two stages: an upstream generation stage and a downstream conditioning stage. 080
081 In the generation stage, we sample three independent external reference im- 081
082 ages per scene (a background, a face portrait, and an object photo), avoiding 082
083 self-reconstruction collages by never cropping references from the ground-truth 083
084 image itself. Each reference is described using Qwen3-VL [1], and the result- 084
085 ing descriptions are composed into a unified scene prompt spanning four sce- 085
086 nario categories (normal, lighting, interaction, and position). These prompts, 086
087 together with the reference images, condition a FLUX.2 model [2] to synthesize 087
088 the ground-truth target image at 1280 × 720 resolution. As a final verification 088
089 step, we use Qwen3-VL [1] as an automatic judge to check whether each sam- 089
090 pled reference is faithfully present in the composed target image; samples in 090
4 ECCV 2026 Submission #8
091 which references are missing or inconsistently rendered are discarded, while the 091
092 remainder are retained for downstream conditioning. Based on empirical insights 092
093 from this verification stage, we cap the number of composed reference objects 093
094 per scene at three, which we find sufficient for reliable presence-checking while 094
095 keeping scene composition tractable. In the downstream stage, we prepare Re- 095
096 fCompose training samples by first extracting RGBA cutouts of each verified 096
097 reference via RMBG-1.4 [4] background removal, then estimating spatial layout 097
098 through a hybrid approach combining Diffusion Features (DIFT) [15] correspon- 098
099 dence matching from a Stable Diffusion [12] U-Net with heuristic bounding-box 099
100 priors as fallback. These cutouts are composited onto a black canvas at their 100
101 estimated locations, subject to area constraints (background ≤ 100%, person 101
102 ≤ 20%, object ≤ 8%), while a monocular depth map is independently predicted 102
103 from the ground-truth image using Depth Anything 3 [10]. The resulting can- 103
104 vas and depth conditioning signals, alongside the text prompt and ground-truth 104
105 image, are exported into a training manifest for LoRA [8] fine-tuning. 105
106 Training schedule. LoRA [8] training proceeds in three stages over 50,000 iter- 106
107 ations at 1280 × 720, as summarised in the main paper: (1) depth-only warm- 107
108 up on a black canvas (10,000 iterations); (2) appearance grounding with crops 108
109 placed at Grounding DINO [11] boxes on ground-truth images (20,000 itera- 109
110 tions); (3) robustness fine-tuning with photometric jitter and mild spatial per- 110
111 turbations (remaining iterations). Rank r=32 low-rank adapters are injected into 111
112 double-stream attention blocks following EasyControl [17]. 112
113 References 113
114 1. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, 114
115 C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., 115
116 Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, 116
117 D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., 117
118 Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, 118
119 T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, 119
120 F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, 120
121 Y., Zhu, K.: Qwen3-VL technical report. arXiv preprint arXiv:2511.21631 (2025) 121
122 2. Black Forest Labs: FLUX.2: Frontier visual intelligence (2025), https://bfl.ai/ 122
123 blog/flux-2 123
124 3. Black Forest Labs: FLUX.2 [dev] (2026), https://huggingface.co/black- 124
125 forest-labs/FLUX.2-dev 125
126 4. BRIA AI: RMBG-1.4: Background removal model (2024), https://huggingface. 126
127 co/briaai/RMBG-1.4 127
128 5. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, 128
129 A.: Emerging properties in self-supervised vision transformers. arXiv preprint 129
130 arXiv:2104.14294 (2021) 130
131 6. Feng, W., Zhu, W., Fu, T.j., Jampani, V., Ganin, Y., Yang, X.E., Wang, W.Y.: 131
132 LayoutGPT: Compositional visual planning and generation with large language 132
133 models. In: Advances in Neural Information Processing Systems (2024) 133
ECCV 2026 Submission #8 5
134 7. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: GANs 134
135 trained by a two time-scale update rule converge to a local Nash equilibrium. 135
136 In: NeurIPS (2017) 136
137 8. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: 137
138 LoRA: Low-rank adaptation of large language models. International Conference on 138
139 Learning Representations (2022) 139
140 9. Li, L.H., Zhang, P., Zhang, H., Yang, J., Li, C., Zhong, Y., Wang, L., Yuan, L., 140
141 Zhang, L., Hwang, J.N., Chang, K.W., Gao, J.: Grounded language-image pre- 141
142 training. In: CVPR. pp. 10965–10975 (2022) 142
143 10. Lin, H., Chen, S., Liew, J.H., Chen, D.Y., Li, Z., Shi, G., Feng, J., Kang, B.: 143
144 Depth anything 3: Recovering the visual space from any views. arXiv preprint 144
145 arXiv:2511.10647 (2025) 145
146 11. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., 146
147 Zhu, J., et al.: Grounding DINO: Marrying DINO with grounded pre-training for 147
148 open-set object detection. arXiv preprint arXiv:2303.05499 (2023) 148
149 12. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution 149
150 image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF 150
151 Conference on Computer Vision and Pattern Recognition. pp. 10684–10695 (2022) 151
152 13. Srivastava, D., Zhang, X., Wen, H., Wen, C., Tu, Z.: Lay-your-scene: Natural scene 152
153 layout generation with diffusion transformers. arXiv preprint arXiv:2505.04718 153
154 (2025) 154
155 14. Tang, H., et al.: CreatiLayout: Siamese multimodal diffusion transformer for cre- 155
156 ative layout-to-image generation. arXiv preprint arXiv:2412.03859 (2024) 156
157 15. Tang, L., Jia, M., Wang, Q., Phoo, C.P., Hariharan, B.: Emergent correspon- 157
158 dence from image diffusion. In: Advances in Neural Information Processing Systems 158
159 (2023) 159
160 16. Xiang, Q., Sun, S., Li, B., Song, D., Li, H., Chen, N., Tang, X., Hu, Y., Zhang, J.: 160
161 InstanceAssemble: Layout-aware image generation via instance assembling atten- 161
162 tion. arXiv preprint arXiv:2509.16691 (2025) 162
163 17. Zhang, Y., et al.: EasyControl: Transfer ControlNet to video diffusion for con- 163
164 trollable generation and editing. In: Proceedings of the IEEE/CVF Conference on 164
165 Computer Vision and Pattern Recognition (2025) 165