import * as Viz from "@viz-js/viz";
import React, { useEffect, useRef, useState } from "react";
import { FaCode, FaImage, FaMagic, FaPaperPlane } from "react-icons/fa";
import { Bounce, ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import {
	requestGraphvizCodeEdit,
	requestGraphvizCodeFromImage,
} from "./api/server";
import { CodePanel } from "./components/CodePanel";
import { GraphSketchpad } from "./components/GraphSketchpad";
import { Header } from "./components/Header";
import { PreviewPanel } from "./components/PreviewPanel";

const ViewTabs = [
	{ id: "code", label: "Code", Icon: FaCode },
	{ id: "preview", label: "Preview", Icon: FaImage },
];

function App() {
	const [theme, setTheme] = useState(() => {
		return localStorage.getItem("theme") || "Light";
	});
	const [loading, setLoading] = useState(false);
	const [graphvizCode, setGraphvizCode] = useState("");
	const [editText, setEditText] = useState("");
	const [validGraphvizImage, setValidGraphvizImage] = useState(false);
	const [error, setError] = useState(null);
	const [useSelectiveChanges, setUseSelectiveChanges] = useState(true);
	const [activeView, setActiveView] = useState("code");

	const graphvizContainerRef = useRef(null);

	useEffect(() => {
		const root = document.documentElement;

		if (theme === "Dark") {
			root.classList.add("dark");
		} else if (theme === "Light") {
			root.classList.remove("dark");
		} else {
			if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
				root.classList.add("dark");
			} else {
				root.classList.remove("dark");
			}
		}

		localStorage.setItem("theme", theme);
	}, [theme]);

	useEffect(() => {
		if (loading) {
			document.body.style.overflow = "hidden";
		} else {
			document.body.style.overflow = "";
		}
		return () => (document.body.style.overflow = "");
	}, [loading]);

	useEffect(() => {
		if (activeView !== "preview") return;

		let cancelled = false;

		const updateGraphPreview = async () => {
			try {
				setError(null);
				const viz = await Viz.instance();
				const svgElement = await viz.renderSVGElement(graphvizCode);

				if (cancelled || !graphvizContainerRef.current) return;

				graphvizContainerRef.current.innerHTML = "";
				graphvizContainerRef.current.appendChild(svgElement);

				setValidGraphvizImage(true);
			} catch (e) {
				if (!cancelled) {
					if (graphvizCode) {
						setError(e.message ?? "Failed to render graph.");
					}

					if (graphvizContainerRef.current) {
						graphvizContainerRef.current.innerHTML = "";
					}

					setValidGraphvizImage(false);
				}
			}
		};

		updateGraphPreview();

		return () => {
			cancelled = true;
		};
	}, [graphvizCode, activeView]);

	const convertToGraphviz = async (pngBlob, useRag) => {
		try {
			setLoading(true);

			const dot = await requestGraphvizCodeFromImage(pngBlob, useRag, 5);
			setGraphvizCode(dot);
			setActiveView("preview");
		} catch (e) {
			toast.error(
				`An unexpected error occurred while converting the sketch: ${e}.`,
			);
			console.error(
				`An unexpected error occurred while converting the sketch: ${e}.`,
			);
		} finally {
			setLoading(false);
		}
	};

	const editGraphvizCode = async () => {
		if (!editText.trim()) return;

		try {
			setLoading(true);

			const dot = await requestGraphvizCodeEdit(
				editText,
				graphvizCode,
				useSelectiveChanges,
			);
			setGraphvizCode(dot);
			setEditText("");
		} catch (e) {
			toast.error(
				`An unexpected error occurred while editing the Graphviz DOT code: ${e}.`,
			);
			console.error(
				`An unexpected error occurred while editing the Graphviz DOT code: ${e}.`,
			);
		} finally {
			setLoading(false);
		}
	};

	const downloadGraphSvg = () => {
		const container = graphvizContainerRef.current;
		if (!container) return;

		const svg = container.querySelector("svg");
		if (!svg) return;

		svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");

		const svgText = new XMLSerializer().serializeToString(svg);
		const blob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
		const url = URL.createObjectURL(blob);

		const a = document.createElement("a");
		a.href = url;
		a.download = "graph.svg";
		document.body.appendChild(a);
		a.click();
		a.remove();

		URL.revokeObjectURL(url);
	};

	const hasCode = Boolean(graphvizCode);

	return (
		<div className="min-h-dvh bg-neutral-50 dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
			{loading && (
				<div
					className="fixed inset-0 z-50 flex items-center justify-center bg-neutral-950/60 backdrop-blur-sm"
					role="dialog"
					aria-modal="true"
					aria-label="Processing">
					<div
						role="status"
						aria-live="polite"
						aria-busy="true"
						className="flex flex-col items-center">
						<svg
							className="h-24 w-24 animate-spin text-neutral-200 dark:text-neutral-700 fill-blue-600"
							viewBox="0 0 100 101"
							fill="none"
							xmlns="http://www.w3.org/2000/svg">
							<path
								d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
								fill="currentColor"
							/>
							<path
								d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
								fill="currentFill"
							/>
						</svg>
						<span className="sr-only">Loading...</span>
					</div>
				</div>
			)}

			<ToastContainer
				position="top-right"
				autoClose={5000}
				hideProgressBar={false}
				newestOnTop={false}
				closeOnClick={false}
				rtl={false}
				pauseOnFocusLoss
				draggable={false}
				pauseOnHover
				theme={theme === "Dark" ? "dark" : "colored"}
				transition={Bounce}
			/>

			<Header theme={theme} setTheme={setTheme} />

			<main className="max-w-7xl mx-auto px-4 md:px-8 py-6">
				<div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
					<section
						aria-label="Sketchpad"
						className="flex flex-col gap-4 min-w-0">
						<div className="flex flex-col gap-1">
							<h2 className="text-xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">
								Sketch your graph
							</h2>
							<p className="text-sm text-neutral-500 dark:text-neutral-400">
								Draw nodes, edges, and labels - or import an image.
							</p>
						</div>

						<GraphSketchpad convertToGraphviz={convertToGraphviz} />
					</section>

					<section aria-label="Output" className="flex flex-col gap-4 min-w-0">
						<div className="flex flex-col gap-1">
							<h2 className="text-xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">
								Review and refine
							</h2>
							<p className="text-sm text-neutral-500 dark:text-neutral-400">
								Inspect the rendered graph, edit the{" "}
								<a
									className="font-bold text-blue-500 underline"
									href="https://graphviz.org/">
									Graphviz DOT
								</a>{" "}
								code, or request AI edits.
							</p>
						</div>

						<div className="flex items-center justify-between gap-2">
							<TabSwitcher
								tabs={ViewTabs}
								active={activeView}
								onChange={setActiveView}
							/>
						</div>

						<div className="relative w-full" style={{ aspectRatio: "1 / 1" }}>
							{activeView === "preview" ? (
								<PreviewPanel
									containerRef={graphvizContainerRef}
									validGraphvizImage={validGraphvizImage}
									onDownload={downloadGraphSvg}
									hasCode={hasCode}
								/>
							) : (
								<CodePanel
									graphvizCode={graphvizCode}
									setGraphvizCode={setGraphvizCode}
									error={error}
								/>
							)}
						</div>

						<EditPrompt
							editText={editText}
							setEditText={setEditText}
							useSelectiveChanges={useSelectiveChanges}
							setUseSelectiveChanges={setUseSelectiveChanges}
							onSubmit={editGraphvizCode}
							disabled={!hasCode}
						/>
					</section>
				</div>
			</main>
		</div>
	);
}

const TabSwitcher = ({ tabs, active, onChange }) => (
	<div
		role="tablist"
		className="inline-flex rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-100 dark:bg-neutral-900 p-1">
		{tabs.map(tab => {
			const isActive = tab.id === active;
			return (
				<button
					key={tab.id}
					role="tab"
					aria-selected={isActive}
					type="button"
					onClick={() => onChange(tab.id)}
					className={`cursor-pointer inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-all ${
						isActive
							? "bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 shadow-sm"
							: "text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100"
					}`}>
					<tab.Icon size={12} />
					{tab.label}
				</button>
			);
		})}
	</div>
);

const EditPrompt = ({
	editText,
	setEditText,
	useSelectiveChanges,
	setUseSelectiveChanges,
	onSubmit,
	disabled,
}) => {
	const canSubmit = !disabled && editText.trim().length > 0;

	const onKeyDown = e => {
		if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && canSubmit) {
			e.preventDefault();
			onSubmit();
		}
	};

	return (
		<div className="flex flex-col gap-2 rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 p-4 focus-within:border-blue-400 dark:focus-within:border-blue-500/60 transition-colors">
			<div className="flex items-start gap-4">
				<FaMagic
					size={14}
					className="mt-2 ml-1 flex-none text-blue-500 dark:text-blue-400"
				/>
				<textarea
					value={editText}
					onChange={e => setEditText(e.target.value)}
					onKeyDown={onKeyDown}
					disabled={disabled}
					rows={2}
					placeholder={
						disabled
							? "Generate code first to request edits…"
							: "Ask for changes — e.g. 'Add a node called Login between Start and Home'"
					}
					className="flex-1 resize-none bg-transparent py-1.5 text-sm text-neutral-800 dark:text-neutral-200 placeholder:text-neutral-400 dark:placeholder:text-neutral-600 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
				/>
			</div>

			<div className="flex items-center justify-between gap-2 pt-2 border-t-2 border-neutral-100 dark:border-neutral-900">
				<label
					className={`inline-flex items-center gap-2 text-sm font-medium text-neutral-600 dark:text-neutral-400 ${
						disabled ? "opacity-60 cursor-not-allowed" : "cursor-pointer"
					}`}
					title="Apply targeted edits instead of regenerating the whole graph">
					<input
						type="checkbox"
						checked={useSelectiveChanges}
						onChange={e => setUseSelectiveChanges(e.target.checked)}
						disabled={disabled}
						className="h-4 w-4 accent-blue-600 cursor-pointer disabled:cursor-not-allowed"
					/>
					Selective edit
				</label>

				<div className="flex items-center gap-4">
					<span className="hidden sm:inline text-xs text-neutral-400 dark:text-neutral-600">
						⌘↵
					</span>
					<button
						type="button"
						onClick={onSubmit}
						disabled={!canSubmit}
						className="cursor-pointer inline-flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:bg-neutral-200 disabled:text-neutral-400 dark:disabled:bg-neutral-800 dark:disabled:text-neutral-600 disabled:hover:cursor-not-allowed">
						<FaPaperPlane size={11} />
						Send
					</button>
				</div>
			</div>
		</div>
	);
};

export default App;
