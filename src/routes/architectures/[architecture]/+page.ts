import { error } from '@sveltejs/kit';
import type { PageLoad } from './$types';
import { architectures } from '~/lib/architecture-navigation';
export const prerender = true;
export const trailingSlash = 'always';
export const load: PageLoad = ({ params }) => {
	const item = architectures.find(
		(item) => item.id === params.architecture && !['dense', 'moe'].includes(item.id)
	);
	if (!item) error(404, 'Unknown transformer architecture');
	return { architecture: item.id };
};
