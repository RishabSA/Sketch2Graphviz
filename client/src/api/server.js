import api from "./axios";

export const fetchGraphvizCode = async (pngBlob, useRag, topKRag) => {
	const form = new FormData();
	form.append("file", pngBlob, "sketch.png");

	const { data } = await api.post("/graphviz_code", form, {
		params: { use_rag: useRag, top_K_rag: topKRag },
	});

	return data;
};
