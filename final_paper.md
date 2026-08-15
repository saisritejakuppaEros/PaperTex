001 RefCompose: Multi-Reference Image Generation 001
002 via LoRA-Conditioned Diffusion 002
003 Anonymous ECCV 2026 Submission 003
004 Paper ID #8 004
005 Abstract. Filmmakers and visual artists routinely need to compose 005
006 multiple references, actors, locations, props, cultural elements, into a 006
007 single coherent shot, but existing tools either fail to scale past a handful 007
008 of references or destroy fine grained subject identity in the process, since 008
009 per reference tokenization scales memory linearly with reference count 009
010 N and generated content often departs from the given references rather 010
011 than reproducing them. We propose RefCompose, a pixel space com- 011
012 positional conditioning framework that decouples where things go from 012
013 what they look like, via a single fixed resolution reference canvas that 013
014 keeps conditioning size constant regardless of reference count. Spatial lay- 014
015 out is induced at inference time from a frozen diffusion transformer and 015
016 extracted via Grounding DINO, requiring no LLM or dedicated layout 016
017 model, while dual stream LoRA adapters inject a layout derived depth 017
018 map and the encoded canvas through separate low rank streams, disen- 018
019 tangling geometric scaffolding from localized appearance. On the Dense 019
020 Layout protocol, RefCompose consistently outperforms layout based and 020
021 state of the art multi reference baselines on color, texture, shape, spatial 021
022 accuracy, and identity/content preservation at higher reference counts, 022
023 all with constant inference memory, making it a practical building block 023
024 for multi subject cinematic composition at production scale. 024
025 Keywords: Diffusion Models · Image Generation · LoRA 025
026 026
Fig. 1: Existing approaches suffer from memory overhead scaling linearly with reference
count and inability to preserve subject appearance without bounding-box supervision.
RefCompose resolves these via a fixed-resolution reference canvas encoding arbitrary
references at constant token cost, and layout induced directly from a structured text
prompt via dense visual cues.
2 ECCV 2026 Submission #8
027 1 Introduction 027
028 Controllable image synthesis has emerged as a central challenge in modern 028
029 generative modelling, with applications spanning advertising, cinematic pre- 029
030 visualisation, and synthetic data production. In the context of movie concept 030
031 generation using text-to-image (T2I) diffusion models, a single composite frame 031
032 serves as the critical blueprint across various stages of production, bridging ab- 032
033 stract concepts and final visual execution. This visual reference must faithfully 033
034 reproduce the appearance of specific actors, real-world locations, cultural aes- 034
035 thetics, and narrative objects [5, 25–27]. 035
036 Existing T2I methods face significant limitations when textual prompts are 036
037 the sole control signal. Text prompts often require iterative refinement to en- 037
038 code texture, identity, and style simultaneously across multiple subjects; yet 038
039 arriving at a prompt that reliably captures all these attributes under heteroge- 039
040 neous data conditions remains a non-trivial challenge. Furthermore, pretrained 040
041 models underrepresent certain concepts (e.g., cultural), leaving gaps in the com- 041
042 mon vocabulary due to inherent bias in the training data distribution. This gap 042
043 undermines the composite image generation through text alone and limits the 043
044 broader applicability of controllable image generation. Layout-to-Image genera- 044
045 tive models [7, 13, 15, 22, 29, 34] use bounding-box layouts to pair spatial coordi- 045
046 nates with localized prompts alongside a global prompt. However, these sparse 046
047 region-level descriptions still fail to reconstruct intended appearances under com- 047
048 plex occlusion scenarios faithfully. These challenges highlight that dense visual 048
049 instance cues provide reliable multimodal content, ensuring subject alignment, 049
050 positioning, and preserving semantic attributes in composite image generation. 050
051 Recent reference-based generation methods scale to multiple subjects, ad- 051
052 dressing cultural and ethnic fidelity gaps left by text-only approaches [8, 14, 052
053 24, 26, 27, 30], but encoding references as independently concatenated token se- 053
054 quences causes token budget to grow linearly, becoming prohibitive beyond four 054
055 to six images, especially at high resolution [5, 26, 27, 30, 31]. Attention-based al- 055
056 ternatives like QKV injection avoid this but remain spatially imprecise, with 056
057 appearance leaking across object boundaries and inversion-based reconstruction 057
058 deviating from the reference even at conservative noise levels [9, 19, 23]. We ob- 058
059 serve that token-scaling stems from the representation, not multi-reference gen- 059
060 eration itself: RefCompose instead compresses any number of reference images 060
061 onto a single fixed-resolution canvas in pixel space, keeping token count constant 061
062 (Figure 1). A coarse prompt-induced layout from the frozen diffusion transformer 062
063 yields a monocular depth map for geometric guidance without extra supervision; 063
064 the canvas and depth map, encoding appearance and structure respectively, are 064
065 injected via a LoRA-based conditioning framework for parameter-efficient adap- 065
066 tation. Our contributions: 066
067 1. RefCompose, a multimodal framework combining spatial compositional 067
068 conditioning and textual prompts for multi-reference generation, via a canvas- 068
069 based multi-reference representation that eliminates token explosion at 069
070 constant memory cost. 070
ECCV 2026 Submission #8 3
071 2. A decoupled structure-appearance conditioning scheme where a 071
072 depth map controls global geometry and the reference canvas injects localized 072
073 appearance, both aligned to the same spatial layout. 073
074 3. A prompt-induced layout, where a generic composition prompt in- 074
075 duces a coarse multi-subject layout from a frozen diffusion transformer with no 075
076 bounding-box supervision or dedicated layout model, and a depth map derived 076
077 from it provides geometric guidance throughout denoising. 077
078 2 Related Work 078
079 Reference-guided image synthesis. Conditioning generation on reference images 079
080 has a long history in diffusion literature [5,8,12,21,22,29,32]. IP-Adapter [31] de- 080
081 couples image and text conditioning via cross-attention, enabling subject-driven 081
082 generation from a single reference, and follow-up methods share a common struc- 082
083 ture of encoding each reference independently and injecting it as an additional 083
084 token sequence [12, 14, 21, 24], so token cost grows with reference count, an issue 084
085 made explicit by architectures that concatenate multiple VAE encodings, which 085
086 overflow GPU memory beyond a small number of references [5, 25–27, 30, 31]. 086
087 Our canvas-based representation avoids this by keeping token count constant 087
088 regardless of reference count. 088
089 Layout-conditioned generation. Spatial control has been pursued via layout pri- 089
090 ors of various kinds [7, 13, 15, 22, 29, 32, 34]: LayoutGPT [7] uses an LLM to 090
091 generate bounding boxes for conditioning, LayoutDiffusion [34] models spatial 091
092 arrangement distributions directly, and GLIGEN [13] injects grounding tokens 092
093 at specified locations via gated self-attention for object-level placement. These 093
094 add pipeline complexity; we find an explicit layout component unnecessary, since 094
095 state-of-the-art DiTs can use a generic composition prompt for coarse layout suf- 095
096 ficient for canvas placement and depth extraction, though comparison against 096
097 specialized layout generators remains future work. 097
098 Depth, structural, and LoRA-based conditioning. ControlNet [32] showed pre- 098
099 computed spatial signals (depth, canny edges, pose) provide strong structural 099
100 guidance for UNet-based diffusion via a trainable parallel encoder, though adapt- 100
101 ing this to DiTs is non-trivial given the architectural shift from UNet. EasyCon- 101
102 trol [33] bridges this gap with a lightweight Condition Injection LoRA integrating 102
103 spatial conditioning into DiT attention via causal attention and KV caching, 103
104 achieving plug-and-play control without retraining; we build on EasyControl 104
105 as the conditioning backbone, using depth maps from layout-induced genera- 105
106 tions for geometric guidance while leaving base weights intact. More broadly, 106
107 LoRA [10] is the standard mechanism for efficient fine-tuning, injecting style, 107
108 identity, and appearance features without modifying base parameters, and offers 108
109 more stable transfer than attention manipulation strategies like QKV injection 109
110 or self-attention swapping, which are sensitive to target object footprint and 110
111 inconsistent for small subjects [9, 19, 23]. We also find inversion-based pipelines 111
4 ECCV 2026 Submission #8
112 fragile: above 0.5 noise strength, structural distortion and fidelity loss become 112
113 severe. EasyControl’s Condition Injection LoRA avoids both failure modes via 113
114 rank-decomposed weight updates rather than activation manipulation. 114
115 Multi-condition generation. Handling multiple conditioning signals is non-trivial 115
116 since signals may conflict in attention layers. Most closely related is Canvas-to- 116
117 Image [6], which performs compositional generation from a multimodal canvas 117
118 but lacks explicit orientation control and degrades in densely cluttered scenes; 118
119 Mixture-of-Experts adapters [28] and composable diffusion [17] address multi- 119
120 signal conditioning via routing or score composition, typically at the cost of 120
121 inference complexity or per-condition fidelity. We instead adopt a disentangled 121
122 design: EasyControl [33] separates spatial and subject signals via position-aware 122
123 training and KV caching, without mutual interference, and our method exploits 123
124 this to jointly inject depth (structure) and canvas (appearance) within a single 124
125 forward pass. 125
126 3 Methodology 126
127 Multi-reference image generation must preserve each reference’s identity while 127
128 keeping conditioning cost fixed regardless of reference count. Token-concatenation 128
129 approaches fail here, since each added reference appends a new encoding se- 129
130 quence, causing linear growth in attention complexity and cross-reference inter- 130
131 ference. We decompose controllable generation into two orthogonal sub-problems: 131
132 where each object appears (structure) and what it looks like (appearance). Struc- 132
133 ture is encoded via a layout-derived geometric scaffold; appearance is grounded 133
134 locally via a fixed-resolution reference canvas. Both signals are injected into a 134
135 frozen diffusion transformer via decoupled LoRA streams, keeping token com- 135
136 plexity constant regardless of reference cardinality. 136
137 Notation. Let R = {I1, . . . , IN } be N RGB references, p an object-centric 137
138 prompt with positional language, and (H, W) the output resolution. Fθ is the 138
frozen diffusion transformer; E, E
−1
139 the VAE encoder/decoder; and r the LoRA 139
140 rank (distinct from the reference set R). 140
141 3.1 Prompt-Induced Layout Generation 141
142 Rather than invoking an external layout model or LLM, we derive spatial struc- 142
143 ture from the frozen diffusion transformer Fθ itself. We restrict p to compo- 143
144 sitional and relational language (scene type, camera framing, and inter-subject 144
145 placement; e.g., “a person on the left, a product on the right”) and delegate pho- 145
146 tometric identity to the reference canvas (Section 3.2). This splits conditioning 146
147 into orthogonal channels: p specifies where subjects belong; the canvas specifies 147
148 what they look like. Text-conditioned flow-matching over T=40 denoising steps 148
149 yields a coarse layout image: 149
 L = \mathcal {F}_\theta (\mathbf {p}) \in \mathbb {R}^{H \times W \times 3}. 150 L = \mathcal {F}_\theta (\mathbf {p}) \in \mathbb {R}^{H \times W \times 3}. (1) 150
ECCV 2026 Submission #8 5
Fig. 2: Overview of RefCompose: prompt-induced layout generation, correspondenceguided canvas construction, and decoupled structure-appearance conditioning via dual
LoRA streams in the MMDiT blocks, preserving full-resolution identity at constant
token complexity.
151 Open-vocabulary detector G (Grounding DINO [18]) localises each reference in 151
152 L, producing boxes that tie references to output regions: 152
153 \mathcal {B} = \mathcal {G}(L) = \{b_1, \ldots , b_N\}, \quad b_k = (x_k, y_k, w_k, h_k), (2) 153
154 where bk is the region assigned to reference Ik. Because depth Ds = Ψ(L) (Sec- 154
155 tion 3.3) and canvas assembly (Section 3.2) both read from the same L and B, 155
156 geometric and appearance conditioning share one spatial frame, avoiding mis- 156
157 alignment that would arise if layout, boxes, and depth were computed indepen- 157
158 dently. 158
159 3.2 Reference Canvas Representation 159
160 Sequential encoding (independently tokenising and concatenating each reference) 160
161 scales linearly with N and forces incompatible token streams at every attention 161
6 ECCV 2026 Submission #8
layer. We instead use a reference canvas C ∈ R
H×W×3
162 : a fixed-resolution image 162
163 in which each reference occupies its expected output region, built in pixel space 163
164 to preserve texture and identity. Each reference is resized and placed at its box: 164
165 165
166 \phi (I_k, b_k) = \mathrm {Resize}(I_k,\; h_k \times w_k)\;\text {placed at}\; b_k. (3) 166
167 When boxes overlap, the nearer reference (by depth, Section 3.3) overwrites the 167
168 farther one. C is encoded once by the VAE: 168
 z_{\mathcal {C}} = \mathcal {E}(\mathcal {C}) \in \mathbb {R}^{(H/f) \times (W/f) \times c}, 169 z_{\mathcal {C}} = \mathcal {E}(\mathcal {C}) \in \mathbb {R}^{(H/f) \times (W/f) \times c}, (4) 169
170 with f the VAE downsampling factor (f = 8 typically) and c the latent channel 170
171 dimension. Thus |zC| = const for all N, versus sequential schemes costing N · 171
172 (H/f)(W/f)c, capping GPU memory independent of reference count. 172
173 3.3 Geometric Scaffolding via Depth Estimation 173
174 Compositing alone lacks depth ordering, surface orientation, and occlusion cues 174
175 needed for 3D plausibility. We estimate a monocular depth map from the layout 175
176 image: 176
 D_s = \Psi (L) \in \mathbb {R}^{H \times W}, 177 D_s = \Psi (L) \in \mathbb {R}^{H \times W}, (5) 177
178 using Depth Anything 3 [16]. Deriving Ds from L (rather than the references) 178
179 ensures L, B, and Ds share an exact spatial frame. Since Ψ runs on generated 179
180 layouts, not real photos, we treat its output as a relative geometric prior, not 180
181 metric ground truth. 181
For each reference k, we compute mean depth ¯ 182 dk = mean(Ds | bk) and sort 182
183 far-to-near as permutation π. The canvas is assembled by sequential overwrite 183
184 compositing: 184
 \mathcal {C} = \mathrm {Paste}\!\left ( \phi (I_{\pi (1)}, b_{\pi (1)}), \ldots , \phi (I_{\pi (N)}, b_{\pi (N)}) \right ), 185 \mathcal {C} = \mathrm {Paste}\!\left ( \phi (I_{\pi (1)}, b_{\pi (1)}), \ldots , \phi (I_{\pi (N)}, b_{\pi (N)}) \right ), (6) 185
186 overwriting overlaps with the nearer reference, a practical ordering signal, not 186
187 an infallible occlusion oracle. 187
188 Ds is VAE-encoded for compatibility with latent denoising: 188
 z_{D_s} = \mathcal {E}(D_s) \in \mathbb {R}^{(H/f) \times (W/f) \times c}, 189 z_{D_s} = \mathcal {E}(D_s) \in \mathbb {R}^{(H/f) \times (W/f) \times c}, (7) 189
190 carrying occlusion order, surface orientation, and spatial extent in a form com- 190
191 mensurate with zC. 191
192 3.4 Decoupled Structure-Appearance (DSA) LoRA 192
193 Naively concatenating zDs and zC lets depth features leak into appearance and 193
194 vice versa. We instead route each through a dedicated low-rank module. 194
For frozen projection W ∈ R
dout×din 195 in the double-stream attention blocks, 195
196 each LoRA branch learns: 196
 \Delta W_m = B_m A_m, \qquad A_m \in \mathbb {R}^{r \times d_{\mathrm {in}}}, \quad B_m \in \mathbb {R}^{d_{\mathrm {out}} \times r}, \quad m \in \{s,a\}, 197 \Delta W_m = B_m A_m, \qquad A_m \in \mathbb {R}^{r \times d_{\mathrm {in}}}, \quad B_m \in \mathbb {R}^{d_{\mathrm {out}} \times r}, \quad m \in \{s,a\}, (8) 197
ECCV 2026 Submission #8 7
with rank r, s the structure branch (zDs
198 ), and a the appearance branch (zC). 198
199 Effective projection: W + ∆Ws + ∆Wa, base W frozen. 199
200 Starting from terminal latent zT , flow-matching sampling yields 200
 \hat {z}_0 = \mathcal {F}_\theta \bigl (\mathbf {p}, z_T, z_{\mathcal {C}}, z_{D_s}; \AdaptW \bigr ), 201 \hat {z}_0 = \mathcal {F}_\theta \bigl (\mathbf {p}, z_T, z_{\mathcal {C}}, z_{D_s}; \AdaptW \bigr ), (9) 201
where p provides semantic grounding, zDs
202 the geometric scaffold, and zC lo- 202
203 calised appearance, and 203
 v_\theta \bigl (z_t, t, \mathbf {p}, z_{\mathcal {C}}, z_{D_s}; \AdaptW \bigr ) 204 (10) 204
205 denotes the learned velocity at step t. Output via VAE decoding: 205
 \hat {I} = \mathcal {E}^{-1}(\hat {z}_0) \in \mathbb {R}^{H \times W \times 3}. 206 \hat {I} = \mathcal {E}^{-1}(\hat {z}_0) \in \mathbb {R}^{H \times W \times 3}. (11) 206
207 For training, let I denote the ground-truth target image and z0 = E(I) its latent 207
208 encoding. We sample noise z1 ∼ N (0, I), timestep t ∼ U[0, 1], and form the 208
209 interpolated latent zt = (1 − t) z0 + t z1. The LoRA adapters are trained by 209
210 regressing vθ onto the constant rectified-flow velocity z1 − z0, i.e. by minimising 210
211 the conditional flow-matching objective 211
 \mathcal {L}_{\text {FM}} = \mathbb {E}_{t,\, z_0,\, z_1} \left [ \left \lVert v_\theta \bigl (z_t, t, \mathbf {p}, z_{\mathcal {C}}, z_{D_s}; \AdaptW \bigr ) - (z_1 - z_0) \right \rVert _2^2 \right ], 212 \mathcal {L}_{\text {FM}} = \mathbb {E}_{t,\, z_0,\, z_1} \left [ \left \lVert v_\theta \bigl (z_t, t, \mathbf {p}, z_{\mathcal {C}}, z_{D_s}; \AdaptW \bigr ) - (z_1 - z_0) \right \rVert _2^2 \right ], (12) 212
213 where only ∆Ws + ∆Wa is updated while Fθ remains frozen. At inference, sam- 213
214 pling starts from zT ∼ N (0, I) and integrates vθ over T=40 steps to obtain zˆ0 214
and decoded output ˆ 215 I. 215
216 4 Experiments 216
217 4.1 Dataset Preparation 217
218 Dataset Construction. We train on ultra high resolution cinematic imagery 218
219 at 1280 × 720 resolution, drawn from a private dataset sourced from movies, 219
220 where most object references are canonicalized to a neutral pose. We comple- 220
221 ment this with a synthetic dataset generated using Flux 2.0 [2], providing diverse 221
222 object positions and orientations so the model learns pose variation while the 222
223 real dataset supplies richer textures; further details are provided in the supple- 223
224 mentary material. Our pipeline proceeds in five stages. First, each movie shot 224
225 is passed through Qwen3-VL [1] to enumerate all objects present in the scene 225
226 (actors, props, and landmarks). Second, Grounding DINO [18] takes these object 226
227 descriptions as text prompts to detect corresponding bounding boxes, retaining 227
228 only shots yielding more than four detections to ensure compositional density. 228
229 Third, each detected crop is fed to Flux 2.0 Dev [3] to synthesize a neutral-pose, 229
230 neutral-lighting canonicalization, removing scene-specific pose and illumination 230
231 variation. Fourth, we filter canonicalized outputs against their original crops us- 231
232 ing DINO [4] cosine similarity, retaining a sample only if similarity exceeds 0.90. 232
233 Finally, retained shots are annotated again with Qwen3-VL, producing both 233
8 ECCV 2026 Submission #8
234 scene-level and crop-level descriptions (Figure 3). This pipeline yields approxi- 234
235 mately 50,000 samples drawn from 1,000 distinct films, together with additional 235
236 synthetic samples. The final training corpus is evenly split between private cin- 236
237 ematic and synthetic samples (50%/50%). 237
238 238
Fig. 3: Overview of the dataset preparation pipeline. Objects are identified using
Qwen3-VL, localized with Grounding DINO, canonicalized to a neutral pose and lighting using Flux 2.0 Dev [3], filtered via DINO [4] similarity, and finally annotated by
Qwen3-VL.
239 4.2 Inference Prompt Template 239
240 We instantiate the layout pipeline of Section 3.1 at 1280 × 720 with Flux 2.0 240
241 Dev [3] as Fθ. For fair comparison, RefCompose and all baselines share one fixed 241
242 template; bracketed slots are populated automatically from Qwen3-VL [1] scene 242
243 and crop descriptions during benchmark evaluation (Section 4.4): 243
244 A [shot type] of a [scene/environment] featuring [primary objects] arranged 244
245 around [central subject], alongside [secondary objects/details] placed naturally 245
246 throughout the scene. Captured from a [camera perspective] using a [lens type], 246
247 with [lighting condition], [mood/style descriptors], and [photorealistic/cinematic 247
248 rendering style]. 248
249 Because box sizes in L are prompt-driven, relative subject prominence can be 249
250 steered through shot type and scale descriptors in the template. When N is large, 250
251 each bk covers a smaller canvas fraction, so ϕ(Ik, bk) downsamples references into 251
252 tighter patches and within-canvas identity detail degrades even though |zC| stays 252
253 fixed; we assign larger relational scale to priority subjects in p so they receive 253
254 proportionally larger boxes in L. 254
255 4.3 RefCompose LoRA Training 255
256 We extend EasyControl [33]’s Condition Injection LoRA to inject depth (struc- 256
257 ture) and a reference-canvas latent zC (appearance) into a frozen Flux 2.0 Dev 257
258 diffusion transformer [3] via separate low-rank updates ∆W = BA with rank 258
259 r=32 in the double-stream attention blocks; causal attention and KV caching 259
260 keep the two streams disentangled, with the subject LoRA loaded before the 260
261 depth LoRA at inference. 261
262 Training runs for 50,000 iterations at 1280 × 720 in three stages while base 262
263 weights remain frozen: (1) 10,000 iterations on depth with a black canvas, initial- 263
264 ising the structure branch before appearance cues are introduced; without this 264
265 warm-up the model tends to use the canvas stream to compensate for geometry, 265
ECCV 2026 Submission #8 9
266 reducing depth/appearance separation; (2) 20,000 iterations with crops placed 266
267 at Grounding DINO [18] boxes on the ground-truth image, teaching appearance 267
268 grounding under clean placements; (3) the remaining iterations add photometric 268
269 jitter and mild spatial perturbations (affine warps, small translations) to crop 269
270 placement, improving robustness to noisier boxes from the inference-time lay- 270
271 out pipeline. Because both branches condition on a single encoded canvas rather 271
272 than per-reference token streams, GPU memory at inference stays approximately 272
273 constant with reference count. 273
274 4.4 Quantitative Analysis 274
275 Benchmark. We evaluate on the Dense Layout test split of InstanceAssem- 275
276 ble [29]: 5,000 held-out composites (∼8.3 annotated objects/image across people, 276
277 props, environments, and product-like assets) with ground-truth boxes and per- 277
278 instance text descriptions. All metrics in Tables 1 and 2 are computed on this 278
279 split unless stated otherwise. 279
280 Evaluation protocol and GPU memory. All methods run on a single NVIDIA 280
281 H100 at 1280 \times 720 using the prompt template from Section 4.2. We select five 281
282 object crops per image from the largest ground-truth boxes by area, canonical- 282
283 ize each via Flux 2.0 Dev (neutral pose/lighting), and generate scene prompts 283
284 with Qwen3-VL [1], ensuring all methods receive identical textual and visual 284
285 information. CreatiLayout and InstanceAssemble [29] receive only Dense Lay- 285
286 out boxes and text; UNO [27], XVerse [5], and RefCompose additionally receive 286
287 the canonicalized reference crops. Among all baselines, UNO and XVerse are 287
288 the most architecturally compatible with RefCompose and achieve the strongest 288
289 reference-image performance, so we benchmark primarily against them. Region- 289
290 wise metrics use object localizations from Grounding DINO [18]. As Table 1 290
291 shows, token-based methods grow in memory with reference count and go out- 291
292 of-memory at N=6 , while RefCompose stays approximately constant via canvas 292
293 compression. Baselines are thus reported at N=5 , with Table 2 also reporting 293
294 RefCompose at N>5 to demonstrate scalability without added memory cost. 294
295 Following LayoutSAM-Eval [29], we report region-wise quality (Spatial, Color, 295
296 Texture, Shape: placement accuracy and appearance/structure fidelity) and global- 296
297 wise quality (PICK [11], CLIP-T [20] for image-text alignment; DINO [4] for 297
298 composition similarity; DINOrefs for reference consistency; CLIP-I [20] for image- 298
299 level similarity). Higher is better throughout. 299
300 Layout-conditioned baselines (CreatiLayout, InstanceAssemble) receive no 300
301 visual reference, so photometric metrics mainly reflect the value of reference 301
302 conditioning; UNO [27] and XVerse [5] are the more architecturally matched 302
303 comparisons. 303
304 Comparison to token-based reference methods. RefCompose outperforms token- 304
305 based methods UNO [27] and XVerse [5], with higher spatial accuracy (0.908 vs. 305
306 0.750), colour fidelity (0.826 vs. 0.500), and reference alignment (DINOrefs: 0.613 306
307 vs. 0.548). Canvas conditioning also holds memory roughly constant, whereas 307
10 ECCV 2026 Submission #8
Table 1: Peak GPU memory (GB) at inference for varying reference count N (batch
size 1; single H100; 1280 × 720). UNO [27] and XVerse [5] scale with reference count
and fail at N=6; RefCompose stays constant via canvas compression. OOM = out-ofmemory.
Method N=2 N=4 N=6
CreatiLayout [22] 34.7 34.8 34.9
InstanceAssemble [29] 35.2 35.3 35.4
UNO [27] 60.1 74.8 OOM
XVerse [5] 59.6 75.3 OOM
RefCompose 69.0 69.0 69.0
Table 2: Quantitative comparison on the Dense Layout test set [29] (5,000 images;
mean 8.3 objects/image). Baselines use five references (UNO/XVerse cannot run beyond this on one H100); the N>5 RefCompose row uses additional references. Higher
is better. Bold = best overall; improvement row compares RefCompose (N>5) against
the strongest reference-image baseline per metric.
Method Region-wise Quality Global-wise Quality
Spatial↑ Color↑ Texture↑ Shape↑ PICK↑ DINO↑ DINOrefs↑ CLIP-I↑ CLIP-T↑
Layout-only baselines
CreatiLayout [22] 0.726 0.487 0.518 0.501 0.222 0.696 0.482 0.727 0.335
InstanceAssemble [29] 0.790 0.590 0.620 0.610 0.221 0.686 0.457 0.703 0.330
Reference-image baselines
UNO [27] 0.750 0.500 0.530 0.520 0.225 0.821 0.548 0.742 0.329
XVerse [5] 0.665 0.421 0.442 0.428 0.218 0.686 0.506 0.731 0.302
RefCompose (N=5) 0.908 0.826 0.854 0.849 0.229 0.959 0.613 0.766 0.327
vs. best reference baseline +0.172 +0.338 +0.336 +0.341 +0.006 +0.150 +0.077 +0.036 0.000
RefCompose (N>5) 0.922 0.838 0.866 0.861 0.231 0.971 0.625 0.778 0.329
308 token concatenation scales linearly with reference count, underscoring the benefit 308
309 of decoupling appearance and layout cues in multi-object generation. 309
310 Comparison to layout-conditioned methods. CreatiLayout and InstanceAssem- 310
311 ble score well on text alignment, with CreatiLayout achieving the top CLIP-T 311
312 (0.335) and InstanceAssemble the higher spatial score (0.790), showing that ex- 312
313 plicit box supervision aids placement even without reference pixels. Their much 313
314 lower colour, texture, and shape scores reflect the absence of reference pixels 314
315 rather than an architectural limitation, since appearance must be derived from 315
316 text alone. RefCompose’s photometric and DINOrefs gains are thus best judged 316
317 against UNO and XVerse, which share the same memory-bounded regime and 317
318 reference crops. 318
319 Global preference and text alignment. PICK scores are tight (0.218 for XVerse 319
320 to 0.229 for RefCompose), making the larger spatial, photometric, and DINOrefs 320
321 gaps more informative. RefCompose’s CLIP-T (0.327) is slightly below UNO 321
322 (0.329) and the layout-only baselines, suggesting text-only alignment is com- 322
323 plementary to, not the primary indicator of, reference-grounded composition 323
324 quality. 324
ECCV 2026 Submission #8 11
Fig. 4: Qualitative comparison of RefCompose against CreatiLayout, InstanceAssemble, UNO, and XVerse across seven multi-subject scenes (ground truth leftmost).
Layout-conditioned baselines hallucinate appearance; token-based methods show subject drift and detail loss under multi-object composition. RefCompose reproduces both
spatial arrangement and photometric identity most faithfully.
325 4.5 Qualitative Analysis 325
326 Figure 4 corroborates the quantitative trends across seven multi-subject scenes. 326
327 CreatiLayout and InstanceAssemble place objects coherently from boxes and 327
328 text but hallucinate colour, texture, and identity, especially for distinctive ma- 328
329 terials where language is a poor substitute for pixels. UNO [27] and XVerse [5] 329
330 transfer some reference appearance yet show subject drift and softened fine detail 330
331 under multi-object layouts in our qualitative examples. RefCompose composites 331
332 references at full resolution on a shared canvas and conditions denoising with an 332
333 aligned depth map, yielding subjects at correct locations with faithful photomet- 333
334 ric detail; overlapping instances are best separated where depth discontinuities 334
335 match canvas boundaries and the dual LoRA streams limit cross-region leakage. 335
336 We do not include Canvas-to-Image [6] in this comparison because its source 336
337 code is not publicly available and it lacks disentangled dual-LoRA control over 337
338 structure and appearance. Our ablation confirms that neither modality alone suf- 338
339 fices, and that merging depth and canvas through a single LoRA stream causes 339
340 cross-signal overwriting that degrades either geometric placement or textural fi- 340
341 delity. The lower CLIP-T score relative to layout-conditioned baselines reflects 341
342 an expected tradeoff: canvas conditioning can shift generation toward reference 342
12 ECCV 2026 Submission #8
343 fidelity at a minor cost to global text alignment, so spatial and DINOrefs met- 343
344 rics are more diagnostic for reference-grounded composition. Residual failures 344
345 appear when prompt-induced layouts imply ambiguous occlusion or implausible 345
346 depth, which bounds geometric plausibility even though the depth prior corrects 346
347 most floating and interpenetration artefacts relative to layout-only baselines. 347
348 Multi-reference flexibility. 348
349 349
Fig. 5: Qualitative example with two different reference sets for the same scene layout.
The prompt-induced canvas and aligned depth map provide shared structural priors,
while the reference images specify subject appearance. RefCompose generates semantically consistent composites for both reference configurations.
350 Figure 5 evaluates two distinct reference sets for the same scene configuration. 350
351 A Flux 2.0 [2]-generated layout image defines where each subject should appear, 351
352 while the estimated depth map supplies additional geometric cues for relative 352
353 orientation and placement. By conditioning jointly on the reference canvas and 353
354 depth prior, RefCompose synthesises coherent outputs for both reference sets, 354
355 preserving the intended spatial arrangement while transferring the appearance 355
356 of each supplied subject. 356
357 Depth-controlled orientation. 357
ECCV 2026 Submission #8 13
358 358
Fig. 6: Depth-controlled generation with identical reference assets. Changing only the
depth map alters object orientation and scene dynamics, from a static sweet-shop
arrangement to a chaotic environment with floating debris, while reference appearance
remains stable.
359 Figure 6 further isolates the role of the depth prior by holding the reference 359
360 assets fixed and varying only the depth map. In the first example, the depth 360
361 map describes a conventional arrangement in which objects are naturally placed 361
362 around a sweet shop. In the second, the depth map is edited to depict a dynamic 362
363 scene with floating debris, airborne bricks, and scattered objects. Despite iden- 363
364 tical reference inputs, the generated results follow the geometry and orientation 364
365 encoded by each depth map, demonstrating that RefCompose can control scene 365
366 dynamics and spatial arrangement while preserving reference identity. 366
367 4.6 Ablation Study 367
368 368
Fig. 7: Ablation study on an indoor multi-object scene. Removing either the reference
canvas or depth prior degrades identity preservation or geometric consistency, while
the full model preserves object appearance and spatial layout by leveraging both conditioning signals.
369 Figure 7 ablates the reference canvas and depth prior (reference crops shown 369
370 above). Removing the canvas yields plausible layouts but hallucinated colours, 370
371 textures, and identities; removing depth retains partial appearance yet degrades 371
372 placement and occlusion (e.g., the bat and bed remain recognizable but are ar- 372
373 ranged implausibly). The full model preserves both appearance and geometry 373
14 ECCV 2026 Submission #8
374 via decoupled LoRA streams; quantitative metrics are reported in the supple- 374
375 mentary material. 375
376 5 Conclusion 376
377 This work addresses multi-reference image generation under a fixed memory 377
378 cost, where token concatenation and QKV injection approaches struggle to pre- 378
379 serve appearance fidelity and spatial precision, and text-only conditioning fails 379
380 to maintain cultural and ethnic fidelity as the number of references and scene 380
381 heterogeneity grow. RefCompose introduces a pixel-space compositional condi- 381
382 tioning framework that encodes an arbitrary number of reference images onto a 382
383 single fixed-resolution canvas, decoupling token complexity from reference cardi- 383
384 nality. Spatial layout is induced from a structured text prompt through a frozen 384
385 diffusion transformer, with Grounding DINO used to extract boxes from the 385
386 generated layout rather than relying on annotated training boxes or a dedicated 386
387 layout-generation model. A dual-stream LoRA injects depth and appearance sig- 387
388 nals through disentangled low-rank adaptation paths, enforcing geometric consis- 388
389 tency and subject identity simultaneously. Quantitative evaluation demonstrates 389
390 that RefCompose consistently outperforms reference-image baselines across ap- 390
391 pearance fidelity and spatial accuracy metrics, while maintaining constant infer- 391
392 ence memory regardless of reference count. 392
393 6 Limitations 393
394 RefCompose depends on a chain of upstream components, including prompt- 394
395 induced layout L, Grounding DINO localisation, and a Depth Anything 3 map, 395
396 whose errors compound when the structured template (Section 4.2) is ambigu- 396
397 ous, when detections are missed or misaligned on generated images, or when 397
398 monocular depth mis-orders overlapping regions on synthetic layouts. Although 398
399 canvas encoding avoids linear token growth with N, each reference is still resized 399
400 to a fixed H × W patch, so large object counts or small assigned boxes atten- 400
401 uate fine-grained identity before encoding, and viewpoint mismatch between L 401
402 and the reference crops (e.g., bird’s-eye depth with lateral portraits) can yield 402
403 contradictory depth and appearance cues. Our quantitative study caps previous 403
404 methods at five references for fair comparison with memory-limited baselines, 404
405 while reporting an additional RefCompose-only N>5 setting. 405
406 References 406
407 1. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, 407
408 C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., 408
409 Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, 409
410 D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., 410
411 Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, 411
ECCV 2026 Submission #8 15
412 T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, 412
413 F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, 413
414 Y., Zhu, K.: Qwen3-VL technical report. arXiv preprint arXiv:2511.21631 (2025) 414
415 2. Black Forest Labs: FLUX.2: Frontier visual intelligence (2025), https://bfl.ai/ 415
416 blog/flux-2 416
417 3. Black Forest Labs: FLUX.2 [dev] (2026), https://huggingface.co/black- 417
418 forest-labs/FLUX.2-dev 418
419 4. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, 419
420 A.: Emerging properties in self-supervised vision transformers. arXiv preprint 420
421 arXiv:2104.14294 (2021) 421
422 5. Chen, B., Zhao, M., Sun, H., Chen, L., Wang, X., Du, K., Wu, X.: XVerse: Consis- 422
423 tent multi-subject control of identity and semantic attributes via DiT modulation. 423
424 arXiv preprint arXiv:2506.21416 (2025) 424
425 6. Dalva, Y., Qian, G.G., Goldenberg, M., Chen, T.S., Aberman, K., Tulyakov, S., 425
426 Yanardag, P., Wang, K.C.J.: Canvas-to-image: Compositional image generation 426
427 with multimodal controls (2025), https://arxiv.org/abs/2511.21691 427
428 7. Feng, W., Zhu, W., Fu, T.j., Jampani, V., Ganin, Y., Yang, X.E., Wang, W.Y.: 428
429 LayoutGPT: Compositional visual planning and generation with large language 429
430 models. In: Advances in Neural Information Processing Systems (2024) 430
431 8. He, Y., et al.: DreamO: Unified image customization with conditions orthogonal- 431
432 ization. arXiv preprint arXiv:2504.16395 (2025) 432
433 9. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: 433
434 Prompt-to-prompt image editing with cross attention control. In: International 434
435 Conference on Learning Representations (2023) 435
436 10. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: 436
437 LoRA: Low-rank adaptation of large language models. International Conference on 437
438 Learning Representations (2022) 438
439 11. Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., Levy, O.: Pick-a-pic: 439
440 An open dataset of user preferences for text-to-image generation. In: Advances in 440
441 Neural Information Processing Systems (2023) 441
442 12. Kumari, N., Zhang, B., Zhang, R., Shechtman, E., Zhu, J.Y.: Multi-concept cus- 442
443 tomization of text-to-image diffusion. In: Proceedings of the IEEE/CVF Confer- 443
444 ence on Computer Vision and Pattern Recognition. pp. 1931–1941 (2023) 444
445 13. Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., Lee, Y.J.: GLIGEN: 445
446 Open-set grounded text-to-image generation. In: Proceedings of the IEEE/CVF 446
447 Conference on Computer Vision and Pattern Recognition. pp. 22511–22521 (2023) 447
448 14. Li, Z., Cao, M., Wang, X., Qi, Z., Cheng, M.M., Shan, Y.: PhotoMaker: 448
449 Customizing realistic human photos via stacked id embedding. arXiv preprint 449
450 arXiv:2312.04461 (2024) 450
451 15. Lian, L., Li, B., Yala, A., Darrell, T.: Llm-grounded diffusion: Enhancing prompt 451
452 understanding of text-to-image diffusion models with large language models. arXiv 452
453 preprint arXiv:2305.13655 (2023) 453
454 16. Lin, H., Chen, S., Liew, J.H., Chen, D.Y., Li, Z., Shi, G., Feng, J., Kang, B.: 454
455 Depth anything 3: Recovering the visual space from any views. arXiv preprint 455
456 arXiv:2511.10647 (2025) 456
457 17. Liu, N., Li, S., Du, Y., Torralba, A., Tenenbaum, J.B.: Compositional visual gen- 457
458 eration with composable diffusion models. In: European Conference on Computer 458
459 Vision. pp. 423–439 (2022) 459
460 18. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., 460
461 Zhu, J., et al.: Grounding DINO: Marrying DINO with grounded pre-training for 461
462 open-set object detection. arXiv preprint arXiv:2303.05499 (2023) 462
16 ECCV 2026 Submission #8
463 19. Mokady, R., Hertz, A., Aberman, K., Pritch, Y., Cohen-Or, D.: Null-text inver- 463
464 sion for editing real images using guided diffusion models. In: Proceedings of the 464
465 IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6228– 465
466 6238 (2023) 466
467 20. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., 467
468 Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable 468
469 visual models from natural language supervision. In: International Conference on 469
470 Machine Learning (2021) 470
471 21. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dream- 471
472 Booth: Fine tuning text-to-image diffusion models for subject-driven generation. 472
473 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recog- 473
474 nition pp. 22500–22510 (2023) 474
475 22. Tang, H., et al.: CreatiLayout: Siamese multimodal diffusion transformer for cre- 475
476 ative layout-to-image generation. arXiv preprint arXiv:2412.03859 (2024) 476
477 23. Tumanyan, N., Geyer, M., Bagon, S., Dekel, T.: Plug-and-play diffusion features 477
478 for text-driven image-to-image translation. In: Proceedings of the IEEE/CVF Con- 478
479 ference on Computer Vision and Pattern Recognition. pp. 1921–1930 (2023) 479
480 24. Wang, Q., Bai, X., Wang, H., Qin, Z., Chen, A.: InstantID: Zero-shot identity- 480
481 preserving generation in seconds. arXiv preprint arXiv:2401.07519 (2024) 481
482 25. Wang, X., et al.: MS-Diffusion: Multi-subject zero-shot image personalization with 482
483 layout guidance. arXiv preprint arXiv:2406.07209 (2024) 483
484 26. Wu, C., Zheng, P., Yan, R., Xiao, S., Luo, X., Wang, Y., Li, W., Jiang, X., Liu, 484
485 Y., Zhou, J., Liu, Z., Xia, Z., Li, C., Deng, H., Wang, J., Luo, K., Zhang, B., Lian, 485
486 D., Wang, X., Wang, Z., Huang, T., Liu, Z.: OmniGen2: Exploration to advanced 486
487 multimodal generation. arXiv preprint arXiv:2506.18871 (2025) 487
488 27. Wu, S., Huang, M., Wu, W., Cheng, Y., Ding, F., He, Q.: Less-to-more generaliza- 488
489 tion: Unlocking more controllability by in-context generation. In: Proceedings of 489
490 the IEEE/CVF International Conference on Computer Vision (2025) 490
491 28. Wu, X., Huang, S., Wei, F.: Mixture of LoRA experts. In: International Conference 491
492 on Learning Representations (2024) 492
493 29. Xiang, Q., Sun, S., Li, B., Song, D., Li, H., Chen, N., Tang, X., Hu, Y., Zhang, J.: 493
494 InstanceAssemble: Layout-aware image generation via instance assembling atten- 494
495 tion. arXiv preprint arXiv:2509.16691 (2025) 495
496 30. Xiao, S., Wang, Y., Zhou, J., Yuan, H., Xing, X., Yan, R., Wang, S., Huang, T., Liu, 496
497 Z.: OmniGen: Unified image generation. arXiv preprint arXiv:2409.11340 (2024) 497
498 31. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: IP-Adapter: Text compatible im- 498
499 age prompt adapter for text-to-image diffusion models. In: Advances in Neural 499
500 Information Processing Systems (2023) 500
501 32. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image 501
502 diffusion models. In: Proceedings of the IEEE/CVF International Conference on 502
503 Computer Vision. pp. 3836–3847 (2023) 503
504 33. Zhang, Y., et al.: EasyControl: Transfer ControlNet to video diffusion for con- 504
505 trollable generation and editing. In: Proceedings of the IEEE/CVF Conference on 505
506 Computer Vision and Pattern Recognition (2025) 506
507 34. Zheng, G., Zhou, X., Li, X., Qi, Z., Shan, Y., Li, X.: LayoutDiffusion: Controllable 507
508 diffusion model for layout-to-image generation. In: Proceedings of the IEEE/CVF 508
509 Conference on Computer Vision and Pattern Recognition. pp. 22490–22499 (2023) 509



It is important to note that reference crops used at evaluation are not raw pixels copied from the target image: as in our training data construction (Section~\ref{sec:dataset}), each crop is canonicalized via Flux~2.0~Dev to a neutral pose, viewpoint, and lighting before placement on the reference canvas, so the model reconstructs identity from an altered rendering rather than the original target pixels. Our evaluation protocol thus mirrors, rather than relaxes, the crop-canonicalize strategy used during training.


