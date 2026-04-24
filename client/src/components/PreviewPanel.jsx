import React from "react";
import { FaDownload, FaImage } from "react-icons/fa";

export const PreviewPanel = ({
	containerRef,
	validGraphvizImage,
	onDownload,
	hasCode,
}) => {
	return (
		<div className="relative flex h-full w-full flex-col overflow-hidden rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950">
			<div className="flex items-center justify-between border-b border-neutral-200 dark:border-neutral-800 px-4 py-2.5">
				<div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-500">
					<span>Preview</span>
				</div>

				<button
					type="button"
					onClick={onDownload}
					disabled={!validGraphvizImage}
					title="Download as SVG"
					className="cursor-pointer inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-sm font-medium text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:cursor-not-allowed transition-colors">
					<FaDownload size={11} />
					SVG
				</button>
			</div>

			<div className="relative flex-1 min-h-0 overflow-auto bg-[radial-gradient(circle_at_center,#f1f5f9_1px,transparent_1px)] bg-size-[16px_16px] dark:bg-[radial-gradient(circle_at_center,#1f2937_1px,transparent_1px)]">
				{!hasCode && (
					<div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center px-6">
						<div className="flex h-10 w-10 items-center justify-center rounded-lg bg-neutral-100 dark:bg-neutral-900 text-neutral-400 dark:text-neutral-600">
							<FaImage size={16} />
						</div>
						<p className="text-sm font-medium text-neutral-600 dark:text-neutral-400">
							No preview yet
						</p>
						<p className="max-w-xs text-xs text-neutral-500 dark:text-neutral-500">
							Your rendered{" "}
							<span className="font-bold text-blue-500">Graphviz</span> graph
							will appear here once the code is generated.
						</p>
					</div>
				)}

				<div
					ref={containerRef}
					className="flex h-full w-full items-center justify-center p-6 [&>svg]:max-w-full [&>svg]:max-h-full [&>svg]:h-auto [&>svg]:w-auto"
				/>
			</div>
		</div>
	);
};

export default PreviewPanel;
