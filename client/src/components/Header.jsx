import React, { useEffect, useRef, useState } from "react";
import {
	FaArrowRight,
	FaDesktop,
	FaGithub,
	FaGlobe,
	FaInfo,
	FaLinkedin,
	FaMoon,
	FaSun,
} from "react-icons/fa";
import { MdOutlineClose } from "react-icons/md";

const ThemeOptions = [
	{ id: "Light", label: "Light", Icon: FaSun },
	{ id: "Dark", label: "Dark", Icon: FaMoon },
	{ id: "System", label: "System", Icon: FaDesktop },
];

const Links = [
	{
		id: "linkedin",
		label: "LinkedIn",
		url: "https://www.linkedin.com/in/rishab-alagharu",
		Icon: FaLinkedin,
		accent: "text-blue-600 dark:text-blue-400",
	},
	{
		id: "website",
		label: "Personal Website",
		url: "https://rishabalagharu.com/",
		Icon: FaGlobe,
		accent: "text-emerald-600 dark:text-emerald-400",
	},
	{
		id: "github",
		label: "GitHub",
		url: "https://github.com/RishabSA",
		Icon: FaGithub,
		accent: "text-neutral-800 dark:text-neutral-200",
	},
];

export const Header = ({ theme, setTheme }) => {
	const [themeDropdownOpen, setThemeDropdownOpen] = useState(false);
	const [isInfoModalOpen, setIsInfoModalOpen] = useState(false);

	const themeMenuRef = useRef(null);

	useEffect(() => {
		if (!themeDropdownOpen) return;

		const onClickOutside = e => {
			if (themeMenuRef.current && !themeMenuRef.current.contains(e.target)) {
				setThemeDropdownOpen(false);
			}
		};

		const onKeyDown = e => {
			if (e.key === "Escape") setThemeDropdownOpen(false);
		};

		window.addEventListener("mousedown", onClickOutside);
		window.addEventListener("keydown", onKeyDown);

		return () => {
			window.removeEventListener("mousedown", onClickOutside);
			window.removeEventListener("keydown", onKeyDown);
		};
	}, [themeDropdownOpen]);

	const ActiveThemeIcon = ThemeOptions.find(t => t.id === theme)?.Icon ?? FaSun;

	return (
		<>
			<header className="sticky top-0 z-40 w-full border-b border-neutral-200 dark:border-neutral-800 bg-neutral-50/80 dark:bg-neutral-950/80 backdrop-blur-md">
				<div className="flex h-16 max-w-7xl mx-auto px-4 md:px-8 items-center justify-between">
					<div className="flex items-center gap-2.5">
						<img
							src="/assets/icon.svg"
							alt="Sketch2Graphviz Icon"
							className="h-12 w-auto"
						/>
						<span className="text-xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">
							Sketch2Graphviz
						</span>
					</div>

					<div className="flex items-center gap-1.5">
						<div ref={themeMenuRef} className="relative">
							<button
								type="button"
								onClick={() => setThemeDropdownOpen(o => !o)}
								aria-haspopup="menu"
								aria-expanded={themeDropdownOpen}
								aria-label={`Theme: ${theme}`}
								title="Theme"
								className="cursor-pointer inline-flex h-9 items-center gap-2 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 px-3 text-sm font-medium text-neutral-700 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors">
								<ActiveThemeIcon size={14} />
								<span className="hidden sm:inline">{theme}</span>
							</button>
							{themeDropdownOpen && (
								<div
									role="menu"
									className="absolute right-0 mt-1.5 min-w-[140px] overflow-hidden rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 shadow-lg shadow-black/5 dark:shadow-black/30">
									{ThemeOptions.map(option => {
										const active = option.id === theme;
										return (
											<button
												key={option.id}
												type="button"
												role="menuitemradio"
												aria-checked={active}
												onClick={() => {
													setTheme(option.id);
													setThemeDropdownOpen(false);
												}}
												className={`cursor-pointer flex w-full items-center gap-2.5 px-3 py-2 text-sm text-left transition-colors ${
													active
														? "bg-blue-50 text-blue-700 dark:bg-blue-500/10 dark:text-blue-300"
														: "text-neutral-700 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-800"
												}`}>
												<option.Icon size={13} />
												{option.label}
											</button>
										);
									})}
								</div>
							)}
						</div>

						<button
							type="button"
							aria-label="About this project"
							title="About"
							onClick={() => setIsInfoModalOpen(true)}
							className="cursor-pointer inline-flex h-9 w-9 items-center justify-center rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors">
							<FaInfo size={13} />
						</button>
					</div>
				</div>
			</header>

			<InfoModal
				open={isInfoModalOpen}
				onClose={() => setIsInfoModalOpen(false)}
			/>
		</>
	);
};

const InfoModal = ({ open, onClose }) => {
	useEffect(() => {
		if (!open) return;

		const onKeyDown = e => {
			if (e.key === "Escape") onClose();
		};

		window.addEventListener("keydown", onKeyDown);
		return () => window.removeEventListener("keydown", onKeyDown);
	}, [open, onClose]);

	return (
		<div
			tabIndex="-1"
			onClick={onClose}
			aria-hidden={!open}
			className={`fixed inset-0 z-50 flex items-center justify-center p-4 transition-opacity duration-200 ${
				open
					? "bg-black/50 backdrop-blur-sm pointer-events-auto opacity-100"
					: "bg-black/0 pointer-events-none opacity-0"
			}`}>
			<div
				role="dialog"
				aria-modal="true"
				aria-labelledby="info-modal-title"
				onClick={e => e.stopPropagation()}
				className={`relative w-full max-w-lg overflow-hidden rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 shadow-2xl transform transition-all duration-200 ${
					open ? "scale-100 opacity-100" : "scale-95 opacity-0"
				}`}>
				<div className="flex items-center justify-between border-b border-neutral-200 dark:border-neutral-800 px-5 py-4">
					<h3
						id="info-modal-title"
						className="text-base font-semibold text-neutral-900 dark:text-neutral-100">
						About Sketch2Graphviz
					</h3>
					<button
						type="button"
						aria-label="Close"
						onClick={onClose}
						className="cursor-pointer inline-flex h-8 w-8 items-center justify-center rounded-lg text-neutral-500 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-800 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors">
						<MdOutlineClose size={20} />
					</button>
				</div>

				<div className="px-5 py-4 space-y-3">
					<p className="text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">
						Sketch2Graphviz converts sketches and images of graphs and
						flowcharts into Graphviz DOT code, using a LoRA fine-tuned{" "}
						<span className="font-medium text-neutral-800 dark:text-neutral-200">
							Llama 3.2 11B Vision
						</span>{" "}
						paired with retrieval-augmented generation over a{" "}
						<span className="font-medium text-neutral-800 dark:text-neutral-200">
							PostgreSQL + PGVector
						</span>{" "}
						store — turning a tedious manual task into a fast one.
					</p>
				</div>

				<div className="border-t border-neutral-200 dark:border-neutral-800 px-5 py-4">
					<p className="mb-3 text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-500">
						Connect
					</p>
					<ul className="space-y-2">
						{Links.map(link => (
							<li key={link.id}>
								<button
									type="button"
									onClick={() => window.open(link.url, "_blank", "noopener")}
									className="cursor-pointer group flex w-full items-center justify-between rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 px-4 py-2.5 text-left transition-colors hover:bg-neutral-50 dark:hover:bg-neutral-800">
									<div className="flex items-center gap-3">
										<link.Icon size={18} className={link.accent} />
										<span className="text-sm font-medium text-neutral-800 dark:text-neutral-200">
											{link.label}
										</span>
									</div>
									<FaArrowRight
										size={12}
										className="text-neutral-400 dark:text-neutral-500 transition-transform group-hover:translate-x-0.5"
									/>
								</button>
							</li>
						))}
					</ul>
				</div>
			</div>
		</div>
	);
};

export default Header;
