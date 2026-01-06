import api from "./axios";

export const requestGraphvizCodeFromImage = async (
	pngBlob,
	useRag,
	topKRag
) => {
	const form = new FormData();
	form.append("file", pngBlob, "sketch.png");

	const { data } = await api.post("/graphviz_code_from_image", form, {
		params: { use_rag: useRag, top_K_rag: topKRag },
	});

	return data;
};

export const requestGraphvizCodeEdit = async (
	editText,
	graphvizCode,
	useSelectiveChanges
) => {
	const { data } = await api.post("/graphviz_code_edit", {
		edit_text: editText,
		graphviz_code: graphvizCode,
		use_selective_changes: useSelectiveChanges,
	});

	return data;
};
