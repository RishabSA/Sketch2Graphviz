import * as Viz from "@viz-js/viz";
import React, { useEffect, useRef, useState } from "react";
import { GraphSketchpad } from "./GraphSketchpad";

const defaultGraphvizCode = `digraph G {
    rankdir=LR;
    A -> B;
    B -> C;
    A -> C [label="shortcut"];
}`;

export const GraphvizRenderer = () => {
	const [graphvizCode, setGraphvizCode] = useState(defaultGraphvizCode);
	const [error, setError] = useState(null);
	const graphvizContainerRef = useRef(null);

	useEffect(() => {
		let cancelled = false;

		const updateGraphPreview = async () => {
			try {
				setError(null);
				const viz = await Viz.instance();
				const svgElement = await viz.renderSVGElement(graphvizCode);

				if (cancelled || !graphvizContainerRef.current) return;

				// Clear old content and add new SVG
				graphvizContainerRef.current.innerHTML = "";
				graphvizContainerRef.current.appendChild(svgElement);
			} catch (e) {
				if (!cancelled) {
					setError(e.message ?? "Failed to render graph.");
					if (graphvizContainerRef.current)
						graphvizContainerRef.current.innerHTML = "";
				}
			}
		};

		updateGraphPreview();

		return () => {
			cancelled = true;
		};
	}, [graphvizCode]);

	return (
		<div className="w-full h-screen flex flex-col md:flex-row gap-4 p-4 bg-neutral-100 dark:bg-neutral-900 rounded-xl text-neutral-900 dark:text-neutral-300">
			<div className="w-full md:w-1/3 flex flex-col">
				<h2 className="mb-4 text-xl font-semibold">Sketch or Upload a Graph</h2>
				<GraphSketchpad
					onConvert={canvas => {
						if (!canvas) return;

						const dataUrl = canvas.toDataURL("image/png");
						console.log("Canvas PNG data URL:", dataUrl);
					}}
				/>
			</div>

			<div className="w-full md:w-1/3 flex flex-col">
				<h2 className="mb-4 text-xl font-semibold">Graphviz Code</h2>
				<textarea
					className="flex-1 bg-neutral-100 dark:bg-neutral-900 border-neutral-200 dark:border-neutral-800 border-2 rounded-xl p-3 font-mono text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-md"
					value={graphvizCode}
					onChange={e => setGraphvizCode(e.target.value)}
					spellCheck={false}
				/>
				{error && (
					<p className="mt-2 text-xs text-red-500">Graphviz Error: {error}</p>
				)}
			</div>

			<div className="w-full md:w-1/3 flex flex-col">
				<h2 className="mb-4 text-xl font-semibold">Graph Preview</h2>
				<div className="flex-1 bg-neutral-100 dark:bg-neutral-900 border-neutral-200 dark:border-neutral-800 border-2 rounded-xl p-3 overflow-auto shadow-md">
					<div
						ref={graphvizContainerRef}
						className="w-full h-full flex items-center justify-center"
					/>
				</div>
			</div>
		</div>
	);
};
