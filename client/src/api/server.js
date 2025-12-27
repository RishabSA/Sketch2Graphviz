import api from "./axios";

// POST /graphviz_code?pngBlob=...
export const fetchGraphvizCode = async pngBlob => {
	const form = new FormData();
	form.append("file", pngBlob, "sketch.png");

	const { data } = await api.post("/graphviz_code", form, {
		headers: { "Content-Type": "multipart/form-data" },
	});

	return data;
};
