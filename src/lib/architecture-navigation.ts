export const architectures = [
	{ id: 'dense', label: 'Dense', title: 'Dense transformer · GPT-2', path: '/dense/' },
	{ id: 'moe', label: 'MoE', title: 'Mixture of Experts · Granite', path: '/' },
	{
		id: 'mla',
		label: 'MLA',
		title: 'DeepSeek-style Multi-head Latent Attention',
		path: '/architectures/mla/'
	},
	{
		id: 'encoder-decoder',
		label: 'Encoder–decoder',
		title: 'Encoder–decoder transformer · T5',
		path: '/architectures/encoder-decoder/'
	},
	{
		id: 'hybrid',
		label: 'Mamba hybrid',
		title: 'Transformer–Mamba hybrid · Zamba2',
		path: '/architectures/hybrid/'
	},
	{
		id: 'vision-language',
		label: 'Vision–language',
		title: 'Vision-language transformer · SmolVLM',
		path: '/architectures/vision-language/'
	},
	{ id: 'vit', label: 'ViT', title: 'Vision Transformer · DeiT', path: '/architectures/vit/' }
] as const;
export type ArchitectureId = (typeof architectures)[number]['id'];
export type RecordedArchitectureId = Exclude<ArchitectureId, 'dense' | 'moe'>;
