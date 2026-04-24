import React, { useCallback, useEffect, useRef, useState } from "react";
import {
	FaCode,
	FaEraser,
	FaFileUpload,
	FaFont,
	FaMousePointer,
	FaPenNib,
	FaRegCircle,
	FaRegSquare,
	FaSave,
	FaTrash,
} from "react-icons/fa";
import { ImArrowUpRight2 } from "react-icons/im";
import {
	Arrow as KonvaArrow,
	Circle as KonvaCircle,
	Image as KonvaImage,
	Line as KonvaLine,
	Rect as KonvaRect,
	Text as KonvaText,
	Layer,
	Stage,
	Transformer,
} from "react-konva";
import { v4 as uuidv4 } from "uuid";

const DrawAction = {
	Select: "select",
	Rectangle: "rectangle",
	Circle: "circle",
	Arrow: "arrow",
	Pen: "freedraw",
	Text: "text",
	Eraser: "eraser",
};

const PaintOptions = [
	{ id: DrawAction.Select, label: "Select", Icon: FaMousePointer },
	{ id: DrawAction.Pen, label: "Pen", Icon: FaPenNib },
	{ id: DrawAction.Rectangle, label: "Rectangle", Icon: FaRegSquare },
	{ id: DrawAction.Circle, label: "Circle", Icon: FaRegCircle },
	{ id: DrawAction.Arrow, label: "Arrow", Icon: ImArrowUpRight2 },
	{ id: DrawAction.Text, label: "Text", Icon: FaFont },
	{ id: DrawAction.Eraser, label: "Eraser", Icon: FaEraser },
];

const PRESET_COLORS = [
	"#0a0a0a",
	"#ef4444",
	"#f97316",
	"#eab308",
	"#22c55e",
	"#3b82f6",
	"#8b5cf6",
	"#ec4899",
];

export const GraphSketchpad = ({ convertToGraphviz }) => {
	const [color, setColor] = useState("#0a0a0a");
	const [drawAction, setDrawAction] = useState(DrawAction.Pen);
	const [selected, setSelected] = useState(null);
	const [strokeWidth, setStrokeWidth] = useState(4);
	const [pens, setPens] = useState([]);
	const [texts, setTexts] = useState([]);
	const [erasers, setErasers] = useState([]);
	const [rectangles, setRectangles] = useState([]);
	const [circles, setCircles] = useState([]);
	const [arrows, setArrows] = useState([]);
	const [image, setImage] = useState(undefined);
	const [stageSize, setStageSize] = useState(768);
	const [isDraggable, setIsDraggable] = useState(
		drawAction === DrawAction.Select,
	);
	const [imageRenderProps, setImageRenderProps] = useState(null);
	const [editingTextId, setEditingTextId] = useState(null);
	const [textEditor, setTextEditor] = useState({ x: 0, y: 0, value: "" });
	const [useRag, setUseRag] = useState(false);

	const wrapperRef = useRef(null);
	const fileRef = useRef(null);
	const stageRef = useRef(null);
	const transformerRef = useRef(null);
	const isPaintRef = useRef(false);
	const currentShapeRef = useRef(undefined);

	const isCanvasEmpty =
		pens.length === 0 &&
		erasers.length === 0 &&
		rectangles.length === 0 &&
		circles.length === 0 &&
		arrows.length === 0 &&
		texts.length === 0 &&
		image == null;

	useEffect(() => {
		setIsDraggable(drawAction === DrawAction.Select);
	}, [drawAction]);

	useEffect(() => {
		if (image) {
			const imgW = image.naturalWidth;
			const imgH = image.naturalHeight;

			const canvasW = stageSize;
			const canvasH = stageSize;

			const scale = Math.min(canvasW / imgW, canvasH / imgH);
			const displayW = Math.round(imgW * scale);
			const displayH = Math.round(imgH * scale);
			const offsetX = Math.round((canvasW - displayW) / 2);
			const offsetY = Math.round((canvasH - displayH) / 2);

			setImageRenderProps({
				x: offsetX,
				y: offsetY,
				width: displayW,
				height: displayH,
			});
		}
	}, [image, stageSize]);

	useEffect(() => {
		const element = wrapperRef.current;
		if (!element) return;

		const resizeObserver = new ResizeObserver(() => {
			const rect = element.getBoundingClientRect();
			setStageSize(Math.round(rect.width));
		});

		resizeObserver.observe(element);
		return () => resizeObserver.disconnect();
	}, []);

	const downloadURI = (uri, name) => {
		const link = document.createElement("a");
		link.download = name;
		link.href = uri;

		document.body.appendChild(link);
		link.click();

		document.body.removeChild(link);
	};

	const exportSketchBlob = async () => {
		if (!stageRef.current) return null;

		const exportSize = 768;
		const pixelRatio = exportSize / stageSize;

		const dataUrl = stageRef.current.toDataURL({
			pixelRatio,
			mimeType: "image/png",
		});

		const result = await fetch(dataUrl);
		return await result.blob();
	};

	const clearSelection = useCallback(() => {
		setSelected(null);

		if (transformerRef.current) {
			transformerRef.current.nodes([]);
			transformerRef.current.getLayer().batchDraw();
		}
	}, []);

	const onImportImageSelect = useCallback(e => {
		const file = e.target.files[0];

		if (file) {
			const imageUrl = URL.createObjectURL(file);
			const img = new window.Image();
			img.onload = () => {
				setImage(img);
				URL.revokeObjectURL(imageUrl);
			};
			img.src = imageUrl;
		}

		e.target.value = "";
	}, []);

	const onImportImageClick = useCallback(() => {
		fileRef.current.click();
	}, []);

	const onDownloadSketchClick = () => {
		const exportSize = 768;

		const currentStageSize = stageRef.current.width();

		const pixelRatio = exportSize / currentStageSize;

		const dataUri = stageRef.current.toDataURL({
			pixelRatio,
			mimeType: "image/png",
		});

		downloadURI(dataUri, "sketch.png");
	};

	const onClear = useCallback(() => {
		setRectangles([]);
		setCircles([]);
		setPens([]);
		setTexts([]);
		setErasers([]);
		setArrows([]);
		setImage(undefined);

		if (transformerRef.current) {
			transformerRef.current.nodes([]);
			if (transformerRef.current.getLayer) {
				transformerRef.current.getLayer().batchDraw();
			}
		}
	}, []);

	const openTextEditor = (id, t) => {
		const container = wrapperRef.current;
		if (!container) return;

		setEditingTextId(id);
		setTextEditor({ x: t.x, y: t.y, value: t.text });
	};

	const commitText = () => {
		if (!editingTextId) return;

		setTexts(prev =>
			prev.map(t =>
				t.id === editingTextId ? { ...t, text: textEditor.value } : t,
			),
		);

		setEditingTextId(null);
	};

	const onStageMouseUp = () => {
		isPaintRef.current = false;
	};

	const onStageMouseDown = useCallback(() => {
		if (drawAction === DrawAction.Select) return;

		isPaintRef.current = true;

		const stage = stageRef.current;
		const pos = stage.getPointerPosition();

		const x = pos.x;
		const y = pos.y;

		const id = uuidv4();
		currentShapeRef.current = id;

		if (drawAction === DrawAction.Text) {
			const id = uuidv4();
			const newText = { id, x, y, text: "", fontSize: 24, fill: color };

			setTexts(prev => [...prev, newText]);
			setSelected({ type: "text", id });

			requestAnimationFrame(() => {
				openTextEditor(id, newText);
			});

			return;
		}

		if (drawAction === DrawAction.Pen) {
			setPens(prev => [
				...prev,
				{ id, points: [x, y], color, width: strokeWidth },
			]);
		} else if (drawAction === DrawAction.Eraser) {
			setErasers(prev => [...prev, { id, points: [x, y], width: strokeWidth }]);
		} else if (drawAction === DrawAction.Circle) {
			setCircles(prev => [
				...prev,
				{ id, radius: 1, x, y, color, width: strokeWidth },
			]);
		} else if (drawAction === DrawAction.Rectangle) {
			setRectangles(prev => [
				...prev,
				{ id, height: 1, width: 1, x, y, color, strokeWidth: strokeWidth },
			]);
		} else if (drawAction === DrawAction.Arrow) {
			setArrows(prev => [
				...prev,
				{ id, points: [x, y, x, y], color, width: strokeWidth },
			]);
		}
	}, [drawAction, color, strokeWidth]);

	const onStageMouseMove = useCallback(() => {
		if (drawAction === DrawAction.Select || !isPaintRef.current) return;

		const stage = stageRef.current;
		const pos = stage.getPointerPosition();

		const x = pos.x;
		const y = pos.y;

		const id = currentShapeRef.current;

		if (drawAction === DrawAction.Pen) {
			setPens(prev =>
				prev.map(s =>
					s.id === id ? { ...s, points: [...s.points, x, y] } : s,
				),
			);
		} else if (drawAction === DrawAction.Eraser) {
			setErasers(prev =>
				prev.map(s =>
					s.id === id ? { ...s, points: [...s.points, x, y] } : s,
				),
			);
		} else if (drawAction === DrawAction.Circle) {
			setCircles(prev =>
				prev.map(c =>
					c.id === id
						? { ...c, radius: ((x - c.x) ** 2 + (y - c.y) ** 2) ** 0.5 }
						: c,
				),
			);
		} else if (drawAction === DrawAction.Rectangle) {
			setRectangles(prev =>
				prev.map(r =>
					r.id === id ? { ...r, height: y - r.y, width: x - r.x } : r,
				),
			);
		} else if (drawAction === DrawAction.Arrow) {
			setArrows(prev =>
				prev.map(a =>
					a.id === id ? { ...a, points: [a.points[0], a.points[1], x, y] } : a,
				),
			);
		}
	}, [drawAction]);

	const onShapeClick = useCallback(
		(e, type) => {
			if (drawAction !== DrawAction.Select) return;

			const node = e.currentTarget;
			const id = node.id();

			setSelected({ type, id });

			if (!transformerRef.current) return;
			transformerRef.current.nodes([node]);
			transformerRef.current.getLayer().batchDraw();
		},
		[drawAction],
	);

	const deleteSelected = useCallback(() => {
		if (!selected) return;

		if (selected.type === "pen") {
			setPens(prev => prev.filter(p => p.id !== selected.id));
		} else if (selected.type === "text") {
			setTexts(prev => prev.filter(t => t.id !== selected.id));
		} else if (selected.type === "rect") {
			setRectangles(prev => prev.filter(r => r.id !== selected.id));
		} else if (selected.type === "circle") {
			setCircles(prev => prev.filter(c => c.id !== selected.id));
		} else if (selected.type === "arrow") {
			setArrows(prev => prev.filter(a => a.id !== selected.id));
		} else if (selected.type === "image") {
			setImage(undefined);
		}

		clearSelection();
	}, [selected, clearSelection]);

	useEffect(() => {
		const onKeyDown = e => {
			if ((e.key === "Delete" || e.key === "Backspace") && selected) {
				e.preventDefault();
				deleteSelected();
			}
		};

		window.addEventListener("keydown", onKeyDown);

		return () => window.removeEventListener("keydown", onKeyDown);
	}, [selected, deleteSelected]);

	const cursorClass =
		drawAction === DrawAction.Select
			? "cursor-default"
			: drawAction === DrawAction.Eraser
				? "cursor-cell"
				: "cursor-crosshair";

	return (
		<div className="flex flex-col gap-3">
			<div
				className={`relative w-full overflow-hidden rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white ${cursorClass}`}
				style={{ aspectRatio: "1 / 1" }}
				ref={wrapperRef}>
				<Stage
					width={stageSize}
					height={stageSize}
					ref={stageRef}
					onMouseUp={onStageMouseUp}
					onMouseDown={onStageMouseDown}
					onMouseMove={onStageMouseMove}>
					<Layer>
						<KonvaRect
							x={0}
							y={0}
							width={stageSize}
							height={stageSize}
							fill="white"
							id="bg"
							onClick={clearSelection}
						/>

						{image && imageRenderProps && (
							<KonvaImage
								image={image}
								x={imageRenderProps.x}
								y={imageRenderProps.y}
								width={imageRenderProps.width}
								height={imageRenderProps.height}
								draggable={isDraggable}
								onClick={e => onShapeClick(e, "image")}
							/>
						)}

						{arrows.map(arrow => (
							<KonvaArrow
								key={arrow.id}
								id={arrow.id}
								points={arrow.points}
								fill={arrow.color}
								stroke={arrow.color}
								strokeWidth={arrow.width}
								onClick={e => onShapeClick(e, "arrow")}
								draggable={isDraggable}
							/>
						))}

						{rectangles.map(rectangle => (
							<KonvaRect
								key={rectangle.id}
								x={rectangle.x}
								y={rectangle.y}
								height={rectangle.height}
								width={rectangle.width}
								stroke={rectangle.color}
								id={rectangle.id}
								strokeWidth={rectangle.strokeWidth}
								onClick={e => onShapeClick(e, "rect")}
								draggable={isDraggable}
							/>
						))}

						{circles.map(circle => (
							<KonvaCircle
								key={circle.id}
								id={circle.id}
								x={circle.x}
								y={circle.y}
								radius={circle.radius}
								stroke={circle.color}
								strokeWidth={circle.width}
								onClick={e => onShapeClick(e, "circle")}
								draggable={isDraggable}
							/>
						))}

						{pens.map(pen => (
							<KonvaLine
								key={pen.id}
								id={pen.id}
								lineCap="round"
								lineJoin="round"
								stroke={pen.color}
								strokeWidth={pen.width}
								points={pen.points}
								onClick={e => onShapeClick(e, "pen")}
								draggable={isDraggable}
							/>
						))}

						{texts.map(t => (
							<KonvaText
								key={t.id}
								id={t.id}
								x={t.x}
								y={t.y}
								text={t.text}
								fontSize={t.fontSize}
								fill={t.fill}
								draggable={isDraggable}
								onClick={e => onShapeClick(e, "text")}
								onDblClick={() => openTextEditor(t.id, t)}
								onDragEnd={e => {
									const { x, y } = e.target.position();
									setTexts(prev =>
										prev.map(tt => (tt.id === t.id ? { ...tt, x, y } : tt)),
									);
								}}
							/>
						))}

						{erasers.map(eraser => (
							<KonvaLine
								key={eraser.id}
								points={eraser.points}
								stroke="black"
								strokeWidth={eraser.width}
								lineCap="round"
								lineJoin="round"
								globalCompositeOperation="destination-out"
								listening={false}
							/>
						))}

						<Transformer ref={transformerRef} />
					</Layer>
				</Stage>

				{editingTextId && (
					<textarea
						autoFocus
						value={textEditor.value}
						placeholder="Type here..."
						onChange={e =>
							setTextEditor(s => ({ ...s, value: e.target.value }))
						}
						onBlur={commitText}
						onKeyDown={e => {
							if (e.key === "Enter" && !e.shiftKey) {
								e.preventDefault();
								commitText();
							}
							if (e.key === "Escape") {
								setEditingTextId(null);
							}
						}}
						className="absolute z-50 rounded-md border border-neutral-300 bg-white p-2 text-sm text-black shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500"
						style={{
							left: textEditor.x,
							top: textEditor.y,
							minWidth: 120,
						}}
					/>
				)}

				{isCanvasEmpty && (
					<div className="pointer-events-none absolute inset-0 flex items-center justify-center">
						<p className="text-sm font-medium text-neutral-300">
							Start sketching your graph
						</p>
					</div>
				)}

				{drawAction === DrawAction.Select && selected && (
					<div className="pointer-events-none absolute left-3 bottom-3 z-20">
						<div className="rounded-md border border-neutral-200 bg-white/95 px-2.5 py-1.5 text-xs font-medium text-neutral-700 shadow-sm backdrop-blur-sm">
							Press{" "}
							<kbd className="rounded bg-neutral-100 px-1 py-0.5 font-mono text-xs text-neutral-700">
								Del
							</kbd>{" "}
							to remove
						</div>
					</div>
				)}
			</div>

			<div className="rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 overflow-hidden mt-1">
				<div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-neutral-100 dark:border-neutral-900">
					<div className="flex items-center gap-0.5">
						{PaintOptions.map(option => {
							const active = option.id === drawAction;
							return (
								<button
									key={option.id}
									type="button"
									title={option.label}
									aria-label={option.label}
									aria-pressed={active}
									onClick={() => setDrawAction(option.id)}
									className={`cursor-pointer inline-flex h-9 w-9 items-center justify-center rounded-md transition-colors ${
										active
											? "bg-blue-600 text-white"
											: "text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800"
									}`}>
									<option.Icon size={14} />
								</button>
							);
						})}
					</div>

					<div className="flex items-center gap-0.5">
						<input
							type="file"
							ref={fileRef}
							onChange={onImportImageSelect}
							className="hidden"
							accept="image/*"
						/>
						<button
							type="button"
							title="Import image"
							aria-label="Import image"
							onClick={onImportImageClick}
							className="cursor-pointer inline-flex h-9 w-9 items-center justify-center rounded-md text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors">
							<FaFileUpload size={13} />
						</button>
						<button
							type="button"
							title="Download sketch as PNG"
							aria-label="Download sketch"
							onClick={onDownloadSketchClick}
							className="cursor-pointer inline-flex h-9 w-9 items-center justify-center rounded-md text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors">
							<FaSave size={13} />
						</button>
						<div className="mx-1 h-8 w-0.5 bg-neutral-200 dark:bg-neutral-800" />
						<button
							type="button"
							title="Clear canvas"
							aria-label="Clear canvas"
							onClick={onClear}
							disabled={isCanvasEmpty}
							className="cursor-pointer inline-flex h-9 w-9 items-center justify-center rounded-md text-neutral-600 dark:text-neutral-300 hover:bg-red-50 dark:hover:bg-red-950/40 hover:text-red-600 dark:hover:text-red-400 disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:cursor-not-allowed disabled:hover:text-neutral-600 dark:disabled:hover:text-neutral-300 transition-colors">
							<FaTrash size={12} />
						</button>
					</div>
				</div>

				<div className="flex min-w-0 flex-col gap-3 px-3 py-3 sm:flex-row sm:items-center">
					<div className="flex flex-wrap items-center gap-x-4 gap-y-2">
						<span className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-500">
							Color
						</span>
						<div className="flex items-center gap-1">
							{PRESET_COLORS.map(c => (
								<button
									key={c}
									type="button"
									onClick={() => setColor(c)}
									title={c}
									aria-label={`Color ${c}`}
									className={`cursor-pointer h-5 w-5 rounded-full border transition-all ${
										color === c
											? "border-neutral-900 dark:border-white ring-2 ring-offset-1 ring-neutral-900 dark:ring-white dark:ring-offset-neutral-950"
											: "border-neutral-200 dark:border-neutral-700 hover:scale-110"
									}`}
									style={{ backgroundColor: c }}
								/>
							))}
							<label
								className="relative h-5 w-5 cursor-pointer overflow-hidden rounded-full border border-neutral-200 dark:border-neutral-700"
								style={{
									background:
										"conic-gradient(from 0deg, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff, #ff00ff, #ff0000)",
								}}
								title="Custom color">
								<input
									type="color"
									value={color}
									onChange={e => setColor(e.target.value)}
									className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
									aria-label="Custom color"
								/>
							</label>
						</div>
					</div>

					<div className="hidden sm:block h-8 w-0.5 bg-neutral-200 dark:bg-neutral-800" />

					<div className="flex min-w-0 flex-1 items-center gap-3">
						<span className="shrink-0 text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-500">
							{drawAction === DrawAction.Eraser ? "Eraser" : "Stroke"}
						</span>
						<input
							type="range"
							min={1}
							max={40}
							value={strokeWidth}
							onChange={e => setStrokeWidth(Number(e.target.value))}
							className="min-w-0 flex-1 accent-blue-600"
							aria-label="Stroke width"
						/>
						<span className="w-8 text-right text-xs font-mono text-neutral-500 dark:text-neutral-400 tabular-nums">
							{strokeWidth}px
						</span>
					</div>
				</div>
			</div>

			<div className="flex items-center gap-2">
				<label
					className="inline-flex items-center gap-2 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 px-3 py-4 text-sm font-medium text-neutral-700 dark:text-neutral-300 cursor-pointer hover:bg-neutral-50 dark:hover:bg-neutral-900 transition-colors"
					title="Use retrieval-augmented generation for higher accuracy">
					<input
						type="checkbox"
						checked={useRag}
						onChange={e => setUseRag(e.target.checked)}
						className="h-4 w-4 accent-blue-600 cursor-pointer"
					/>
					Use RAG
				</label>

				<button
					type="button"
					onClick={async () => {
						const blob = await exportSketchBlob();
						await convertToGraphviz(blob, useRag);
					}}
					disabled={isCanvasEmpty}
					className="cursor-pointer flex-1 inline-flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-4 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:bg-neutral-200 disabled:text-neutral-400 dark:disabled:bg-neutral-900 dark:disabled:text-neutral-600 disabled:hover:cursor-not-allowed">
					<FaCode size={14} />
					Convert to Graphviz
				</button>
			</div>
		</div>
	);
};

export default GraphSketchpad;
