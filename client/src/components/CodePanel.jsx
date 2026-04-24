import React, { useEffect, useRef, useState } from "react";
import { FaCheck, FaCode, FaCopy } from "react-icons/fa";
import { toast } from "react-toastify";

export const CodePanel = ({ graphvizCode, setGraphvizCode, error }) => {
	const [copied, setCopied] = useState(false);
	const copiedTimerRef = useRef(null);

	useEffect(() => {
		return () => {
			if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
		};
	}, []);

	const handleCopy = async () => {
		if (!graphvizCode) return;

		try {
			await navigator.clipboard.writeText(graphvizCode);

			setCopied(true);
			if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);

			copiedTimerRef.current = setTimeout(() => {
				setCopied(false);
				copiedTimerRef.current = null;
			}, 2000);
		} catch (err) {
			toast.error(`Unable to copy to clipboard: ${err}`);
			console.error(`Unable to copy to clipboard: ${err}`);
		}
	};

	const hasCode = Boolean(graphvizCode);

	return (
		<div className="relative flex h-full w-full flex-col overflow-hidden rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950">
			<div className="flex items-center justify-between border-b border-neutral-200 dark:border-neutral-800 px-4 py-2.5">
				<div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-500">
					<span>graph.dot</span>
				</div>

				<button
					type="button"
					onClick={handleCopy}
					disabled={!hasCode}
					className="cursor-pointer inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-sm font-medium text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:cursor-not-allowed transition-colors">
					{copied ? (
						<FaCheck size={12} className="text-emerald-500" />
					) : (
						<FaCopy size={12} />
					)}
					{copied ? "Copied" : "Copy"}
				</button>
			</div>

			<div className="relative flex-1 min-h-0">
				{!hasCode ? (
					<div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center px-6">
						<div className="flex h-10 w-10 items-center justify-center rounded-lg bg-neutral-100 dark:bg-neutral-900 text-neutral-400 dark:text-neutral-600">
							<FaCode size={16} />
						</div>
						<p className="text-sm font-medium text-neutral-600 dark:text-neutral-400">
							No code yet
						</p>
						<p className="max-w-xs text-xs text-neutral-500 dark:text-neutral-500">
							Sketch a graph and click{" "}
							<span className="font-bold text-blue-500">Convert</span>. The
							generated{" "}
							<span className="font-bold text-blue-500">Graphviz DOT</span> code
							will appear here.
						</p>
					</div>
				) : (
					<textarea
						className="h-full w-full resize-none bg-transparent p-4 font-mono text-sm leading-relaxed text-neutral-800 dark:text-neutral-200 focus:outline-none focus:ring-0"
						value={graphvizCode}
						onChange={e => setGraphvizCode(e.target.value)}
						spellCheck={false}
						placeholder="digraph { ... }"
					/>
				)}

				{error && hasCode && (
					<div className="pointer-events-none absolute left-3 right-3 bottom-3">
						<div className="pointer-events-auto rounded-lg border border-red-200 dark:border-red-900/60 bg-red-50/95 dark:bg-red-950/60 px-3 py-2 text-sm font-medium text-red-700 dark:text-red-300 backdrop-blur-sm">
							<span className="font-semibold">Graphviz error: </span>
							{error}
						</div>
					</div>
				)}
			</div>
		</div>
	);
};

export default CodePanel;
