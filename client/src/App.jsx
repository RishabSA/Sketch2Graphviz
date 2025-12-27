import * as Viz from "@viz-js/viz";
import React, { useEffect, useRef, useState } from "react";
import {
	FaArrowRight,
	FaCheck,
	FaCopy,
	FaDesktop,
	FaDownload,
	FaGithub,
	FaGlobe,
	FaInfo,
	FaLinkedin,
	FaMoon,
	FaSun,
} from "react-icons/fa";
import { MdOutlineClose } from "react-icons/md";
import { Bounce, ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { fetchGraphvizCode } from "./api/server";
import { GraphSketchpad } from "./components/GraphSketchpad";

function App() {
	const [theme, setTheme] = useState(() => {
		return localStorage.getItem("theme") || "Light";
	});
	const [themeDropdownOpen, setThemeDropdownOpen] = useState(false);
	const [loading, setLoading] = useState(false);
	const [isInfoModalOpen, setIsInfoModalOpen] = useState(false);
	const [graphvizCode, setGraphvizCode] = useState("");
	const [validGraphvizImage, setValidGraphvizImage] = useState(false);
	const [copied, setCopied] = useState(false);
	const [error, setError] = useState(null);

	const graphvizContainerRef = useRef(null);
	const copiedTimerRef = useRef(null);

	useEffect(() => {
		return () => {
			if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
		};
	}, []);

	// Apply the theme whenever it changes
	useEffect(() => {
		const root = document.documentElement;

		if (theme === "Dark") {
			root.classList.add("dark");
		} else if (theme === "Light") {
			root.classList.remove("dark");
		} else {
			// System
			if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
				root.classList.add("dark");
			} else {
				root.classList.remove("dark");
			}
		}

		localStorage.setItem("theme", theme);
	}, [theme]);

	// Prevent background scroll while loading
	useEffect(() => {
		if (loading) {
			document.body.style.overflow = "hidden";
		} else {
			document.body.style.overflow = "";
		}
		return () => (document.body.style.overflow = "");
	}, [loading]);

	useEffect(() => {
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
	}, [graphvizCode]);

	const handleCopyGraphvizCode = async () => {
		try {
			await navigator.clipboard.writeText(graphvizCode);

			setCopied(true);
			if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);

			copiedTimerRef.current = setTimeout(() => {
				setCopied(false);
				copiedTimerRef.current = null;
			}, 2000);
		} catch (err) {
			console.error("Failed to copy text: ", err);
		}
	};

	const convertToGraphviz = async pngBlob => {
		try {
			setLoading(true);

			const dot = await fetchGraphvizCode(pngBlob);
			setGraphvizCode(dot);
		} catch (e) {
			toast.error(
				`An unexpected error occurred while attempting to convert the sketch: ${e}.`
			);
		} finally {
			setLoading(false);
		}
	};

	return (
		<div className="mb-24">
			{loading && (
				<div
					className="fixed inset-0 z-50 grid place-items-center bg-black/50 backdrop-blur-xs"
					role="dialog"
					aria-modal="true"
					aria-label="Loading">
					<div role="status" aria-live="polite" aria-busy="true">
						<svg
							className="inline w-20 h-20 text-neutral-200 animate-spin dark:text-neutral-600 fill-blue-600"
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
			<div
				tabIndex="-1"
				onClick={() => setIsInfoModalOpen(false)}
				className={`fixed inset-0 z-50 flex items-center justify-center transition-opacity duration-300 ${
					isInfoModalOpen
						? "bg-black/50 pointer-events-auto opacity-100"
						: "bg-black/0 pointer-events-none opacity-0"
				}`}>
				<div
					onClick={e => e.stopPropagation()}
					className={`relative p-4 w-full max-w-xs md:max-w-2xl max-h-full bg-white rounded-lg shadow-sm dark:bg-neutral-800 transform transition-transform duration-300 ${
						isInfoModalOpen ? "scale-100 opacity-100" : "scale-95 opacity-0"
					}`}>
					<div className="flex items-center justify-between p-4 md:p-5 border-b rounded-t dark:border-neutral-600 border-neutral-200">
						<h3 className="flex items-center text-xl font-semibold text-neutral-900 dark:text-neutral-300">
							<FaInfo
								size={24}
								className="stroke-current text-neutral-500 dark:text-neutral-300 mr-4"
							/>
							Info
						</h3>
						<button
							aria-label="Close"
							className="cursor-pointer text-neutral-400 bg-transparent hover:bg-neutral-200 hover:text-neutral-900 rounded-lg text-sm w-8 h-8 ms-auto inline-flex justify-center items-center dark:hover:bg-neutral-600 dark:hover:text-white"
							onClick={() => setIsInfoModalOpen(false)}>
							<MdOutlineClose
								size={24}
								className="stroke-current text-neutral-500 dark:text-neutral-400"
							/>
						</button>
					</div>
					<div className="p-4 md:p-5 space-y-4">
						<p className="text-base leading-relaxed text-neutral-500 dark:text-neutral-400">
							Sketch2Graphviz is an AI-powered tool that converts an image of a
							graph into working graphviz code, speeding up system design
							processes, architecture design, and making your work a lot easier!
							The model uses a LoRA fine-tuned Llama 3.2 Vision 11B Instruct
							model and Retrieval-Augmented Generation (RAG) with a vector
							database built with PostgreSQL and PGVector.
						</p>
						<p className="text-base leading-relaxed text-neutral-500 dark:text-neutral-400">
							See my personal links below to learn more about me and my work or
							get in contact with me!
						</p>
						<ul className="space-y-4 mb-4">
							<li>
								<label
									onClick={() =>
										window.open(
											"https://www.linkedin.com/in/rishab-alagharu",
											"_blank"
										)
									}
									className="transition-all inline-flex items-center justify-between w-full px-4 py-2 text-neutral-900 bg-white border border-neutral-200 rounded-lg cursor-pointer dark:hover:text-neutral-300 dark:border-neutral-500 hover:text-neutral-900 hover:bg-neutral-100 dark:text-white dark:bg-neutral-800 dark:hover:bg-neutral-700">
									<div className="flex items-center space-x-4">
										<FaLinkedin
											size={24}
											className="stroke-current text-blue-700 dark:text-blue-500"
										/>
										<div className="w-full text-lg font-semibold">LinkedIn</div>
									</div>
									<FaArrowRight
										size={20}
										className="stroke-current text-neutral-500 dark:text-neutral-400"
									/>
								</label>
							</li>
							<li>
								<label
									onClick={() =>
										window.open("https://rishabalagharu.com/", "_blank")
									}
									className="transition-all inline-flex items-center justify-between w-full px-4 py-2 text-neutral-900 bg-white border border-neutral-200 rounded-lg cursor-pointer dark:hover:text-neutral-300 dark:border-neutral-500 hover:text-neutral-900 hover:bg-neutral-100 dark:text-white dark:bg-neutral-800 dark:hover:bg-neutral-700">
									<div className="flex items-center space-x-4">
										<FaGlobe
											size={24}
											className="stroke-current text-green-500 dark:text-green-400"
										/>
										<div className="w-full text-lg font-semibold">
											Personal Website
										</div>
									</div>
									<FaArrowRight
										size={20}
										className="stroke-current text-neutral-500 dark:text-neutral-400"
									/>
								</label>
							</li>
							<li>
								<label
									onClick={() =>
										window.open("https://github.com/RishabSA", "_blank")
									}
									className="transition-all inline-flex items-center justify-between w-full px-4 py-2 text-neutral-900 bg-white border border-neutral-200 rounded-lg cursor-pointer dark:hover:text-neutral-300 dark:border-neutral-500 hover:text-neutral-900 hover:bg-neutral-100 dark:text-white dark:bg-neutral-800 dark:hover:bg-neutral-700">
									<div className="flex items-center space-x-4">
										<FaGithub
											size={24}
											className="stroke-current text-neutral-800 dark:text-neutral-300"
										/>

										<div className="w-full text-lg font-semibold">Github</div>
									</div>
									<FaArrowRight
										size={20}
										className="stroke-current text-neutral-500 dark:text-neutral-400"
									/>
								</label>
							</li>
						</ul>
					</div>
				</div>
			</div>

			<div className="h-screen bg-neutral-100 dark:bg-neutral-900 flex flex-col px-8 py-2">
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
					theme="colored"
					transition={Bounce}
				/>

				<div className="mt-4 mb-8 md:flex md:space-x-20 items-center justify-center">
					<div className="flex items-center space-x-4 justify-center">
						<img
							src="/assets/vite.svg"
							alt="Sketch2Graphviz Icon"
							className="h-10 w-auto"
						/>
						<h1 className="text-3xl font-semibold text-neutral-800 dark:text-neutral-200">
							Sketch2Graphviz
						</h1>
					</div>

					<div className="flex items-center space-x-4 mt-5 md:mt-0 justify-between">
						<div>
							<button
								onClick={() => setThemeDropdownOpen(o => !o)}
								className="cursor-pointer text-neutral-700 dark:text-neutral-300 bg-white dark:bg-neutral-900 border-2 border-neutral-200 dark:border-neutral-600 hover:bg-neutral-200 dark:hover:bg-neutral-800 font-medium rounded-lg text-sm px-5 py-2.5 inline-flex items-center">
								{theme === "Light" && <FaSun size={16} className="mr-2" />}
								{theme === "Dark" && <FaMoon size={16} className="mr-2" />}
								{theme === "System" && <FaDesktop size={16} className="mr-2" />}
								<span>{theme}</span>
							</button>
							{themeDropdownOpen && (
								<div className="transition-all duration-300 ease-in-out absolute z-10 bg-white shadow-md dark:bg-neutral-800 rounded-lg">
									<ul className="py-2 text-sm text-neutral-600 dark:text-neutral-400">
										<li>
											<button
												onClick={() => {
													setTheme("Light");
													setThemeDropdownOpen(false);
												}}
												className="cursor-pointer w-full px-4 py-2 text-left hover:bg-neutral-100 dark:hover:bg-neutral-700 flex items-center">
												<FaSun size={16} className="mr-2" />
												Light
											</button>
										</li>
										<li>
											<button
												onClick={() => {
													setTheme("Dark");
													setThemeDropdownOpen(false);
												}}
												className="cursor-pointer w-full px-4 py-2 text-left hover:bg-neutral-100 dark:hover:bg-neutral-700 flex items-center">
												<FaMoon size={16} className="mr-2" />
												Dark
											</button>
										</li>
										<li>
											<button
												onClick={() => {
													setTheme("System");
													setThemeDropdownOpen(false);
												}}
												className="cursor-pointer w-full px-4 py-2 text-left hover:bg-neutral-100 dark:hover:bg-neutral-700 flex items-center">
												<FaDesktop size={16} className="mr-2" />
												System
											</button>
										</li>
									</ul>
								</div>
							)}
						</div>
						<button
							aria-label="Open information panel"
							title="Info"
							className="cursor-pointer h-10 w-10 text-neutral-500 dark:text-neutral-300 bg-white dark:bg-neutral-900 border-2 border-neutral-200 dark:border-neutral-600 hover:bg-neutral-200 dark:hover:bg-neutral-800 flex items-center justify-center rounded-lg transition-colors"
							onClick={() => setIsInfoModalOpen(true)}>
							<div className="flex items-center">
								<FaInfo
									size={16}
									className="stroke-current text-neutral-500 dark:text-neutral-300"
								/>
							</div>
						</button>
					</div>
				</div>

				<div className="w-full h-full flex flex-col md:flex-row gap-4 bg-neutral-100 dark:bg-neutral-900 rounded-xl text-neutral-900 dark:text-neutral-300">
					<GraphSketchpad convertToGraphviz={convertToGraphviz} />
					<div className="w-full md:w-1/3 flex flex-col min-h-0 mt-6 md:mt-0 gap-4">
						<h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
							Generated Graphviz Code
						</h2>

						<div
							className="relative w-full overflow-hidden rounded-xl border-2 border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900"
							style={{ aspectRatio: "1 / 1" }}>
							<textarea
								className="h-full w-full font-mono p-3 text-sm resize-none bg-transparent focus:outline-none focus:ring-2 focus:ring-blue-500"
								value={graphvizCode}
								onChange={e => setGraphvizCode(e.target.value)}
								spellCheck={false}
							/>
							{error && (
								<div className="absolute left-3 bottom-3">
									<p className="text-xs text-red-500 bg-white/90 dark:bg-neutral-900/90 px-2 py-1 rounded-md border border-red-200 dark:border-red-900">
										Graphviz Error: {error}
									</p>
								</div>
							)}
						</div>

						<button
							type="button"
							onClick={handleCopyGraphvizCode}
							disabled={!graphvizCode}
							className="w-fit cursor-pointer flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-4 text-sm font-bold text-neutral-100 transition-all hover:bg-blue-700 disabled:bg-neutral-300 disabled:dark:bg-neutral-800 disabled:dark:text-neutral-500 disabled:hover:cursor-not-allowed">
							{copied ? <FaCheck size={16} /> : <FaCopy size={16} />}
							{copied ? "Copied!" : "Copy Graphviz Code"}
						</button>
					</div>

					<div className="w-full md:w-1/3 flex flex-col min-h-0 mt-6 md:mt-0 gap-4">
						<h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
							Graphviz Preview
						</h2>

						<div
							className="w-full overflow-auto rounded-xl border-2 border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-3"
							style={{ aspectRatio: "1 / 1" }}>
							<div
								ref={graphvizContainerRef}
								className="w-full h-full flex items-center justify-center"
							/>
						</div>

						<button
							type="button"
							disabled={!validGraphvizImage}
							className="w-fit cursor-pointer flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-4 text-sm font-bold text-neutral-100 transition-all hover:bg-blue-700 disabled:bg-neutral-300 disabled:dark:bg-neutral-800 disabled:dark:text-neutral-500 disabled:hover:cursor-not-allowed">
							<FaDownload size={16} />
							Download Graphviz Graph
						</button>
					</div>
				</div>
			</div>
		</div>
	);
}

export default App;
