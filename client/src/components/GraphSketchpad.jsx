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
	{
		id: DrawAction.Select,
		label: "Select",
		icon: <FaMousePointer size={16} />,
	},
	{
		id: DrawAction.Rectangle,
		label: "Rectangle",
		icon: <FaRegSquare size={16} />,
	},
	{
		id: DrawAction.Circle,
		label: "Circle",
		icon: <FaRegCircle size={16} />,
	},
	{
		id: DrawAction.Arrow,
		label: "Arrow",
		icon: <ImArrowUpRight2 size={16} />,
	},
	{
		id: DrawAction.Pen,
		label: "Pen",
		icon: <FaPenNib size={16} />,
	},
	{
		id: DrawAction.Text,
		label: "Text",
		icon: <FaFont size={16} />,
	},
	{ id: DrawAction.Eraser, label: "Eraser", icon: <FaEraser size={16} /> },
];

export const GraphSketchpad = ({ convertToGraphviz }) => {
	const [color, setColor] = useState("#000000");
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
		drawAction === DrawAction.Select
	);
	const [imageRenderProps, setImageRenderProps] = useState(null);
	const [editingTextId, setEditingTextId] = useState(null);
	const [textEditor, setTextEditor] = useState({ x: 0, y: 0, value: "" });

	const wrapperRef = useRef(null);
	const fileRef = useRef(null);
	const stageRef = useRef(null);
	const transformerRef = useRef(null);
	const isPaintRef = useRef(false);
	const currentShapeRef = useRef(undefined);

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

		// Position textarea relative to the stage container
		setEditingTextId(id);
		setTextEditor({ x: t.x, y: t.y, value: t.text });
	};

	const commitText = () => {
		if (!editingTextId) return;

		setTexts(prev =>
			prev.map(t =>
				t.id === editingTextId ? { ...t, text: textEditor.value } : t
			)
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

			// open editor immediately
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
				prev.map(s => (s.id === id ? { ...s, points: [...s.points, x, y] } : s))
			);
		} else if (drawAction === DrawAction.Eraser) {
			setErasers(prev =>
				prev.map(s => (s.id === id ? { ...s, points: [...s.points, x, y] } : s))
			);
		} else if (drawAction === DrawAction.Circle) {
			setCircles(prev =>
				prev.map(c =>
					c.id === id
						? { ...c, radius: ((x - c.x) ** 2 + (y - c.y) ** 2) ** 0.5 }
						: c
				)
			);
		} else if (drawAction === DrawAction.Rectangle) {
			setRectangles(prev =>
				prev.map(r =>
					r.id === id ? { ...r, height: y - r.y, width: x - r.x } : r
				)
			);
		} else if (drawAction === DrawAction.Arrow) {
			setArrows(prev =>
				prev.map(a =>
					a.id === id ? { ...a, points: [a.points[0], a.points[1], x, y] } : a
				)
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
		[drawAction]
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

	return (
		<div className="w-full md:flex-1 md:w-1/3 flex flex-col">
			<h2 className="mb-4 text-xl font-semibold text-neutral-900 dark:text-neutral-100">
				Sketch your Graph
			</h2>
			<div className="w-full max-w-full mx-auto">
				<div
					className={`relative w-full overflow-hidden rounded-xl border-2 border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 hover:cursor-crosshair ${
						drawAction === DrawAction.Eraser
							? "cursor-not-allowed"
							: "cursor-crosshair"
					}`}
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
											prev.map(tt => (tt.id === t.id ? { ...tt, x, y } : tt))
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
							className="absolute z-50 border border-neutral-300 rounded-md p-2 bg-white text-black"
							style={{
								left: textEditor.x,
								top: textEditor.y,
								minWidth: 120,
							}}
						/>
					)}

					{drawAction === DrawAction.Select && selected && (
						<div className="absolute left-3 bottom-3 z-20 pointer-events-none">
							<div className="rounded-md border border-neutral-200 dark:border-neutral-700 bg-white/90 dark:bg-neutral-900/90 backdrop-blur px-3 py-2 text-xs text-neutral-700 dark:text-neutral-200 shadow-sm">
								Tip: Press <span className="font-semibold">Delete</span> to
								delete the selection
							</div>
						</div>
					)}
				</div>
			</div>
			<div className="mt-4 w-full flex flex-col items-center gap-3">
				<div className="w-full flex flex-wrap gap-2 items-center justify-center">
					<div className="flex items-center justify-center">
						<div className="grid grid-cols-4 items-center overflow-hidden rounded-2xl border-2 border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900">
							{PaintOptions.map(({ id, label, icon }) => {
								const active = id === drawAction;
								return (
									<button
										key={id}
										type="button"
										title={label}
										onClick={() => setDrawAction(id)}
										className={`flex items-center justify-center px-4 py-4 text-md font-semibold transition-all duration-300 cursor-pointer ${
											active
												? "bg-blue-600 text-white"
												: "bg-white text-neutral-900 hover:bg-neutral-200 dark:bg-neutral-900 dark:text-neutral-100 dark:hover:bg-neutral-800"
										}`}>
										{icon}
									</button>
								);
							})}

							<button
								type="button"
								onClick={onClear}
								title="Clear"
								className="flex items-center justify-center px-4 py-4 text-md font-semibold transition-all duration-300 cursor-pointer bg-white text-neutral-900 hover:bg-neutral-200 dark:bg-neutral-900 dark:text-neutral-100 dark:hover:bg-neutral-800">
								<FaTrash size={16} />
							</button>
						</div>
					</div>

					<div className="flex flex-col items-center justify-center">
						<input
							type="file"
							ref={fileRef}
							onChange={onImportImageSelect}
							className="hidden"
							accept="image/*"
						/>

						<button
							type="button"
							title="Import Image"
							onClick={onImportImageClick}
							className="cursor-pointer flex items-center justify-center gap-2 rounded-t-xl bg-white px-4 py-4 text-sm font-bold text-neutral-900 transition-all hover:bg-neutral-200 dark:bg-neutral-900 dark:text-neutral-100 dark:hover:bg-neutral-800 border-2 border-neutral-200 dark:border-neutral-800">
							<FaFileUpload size={16} />
						</button>

						<button
							type="button"
							title="Download Sketch"
							onClick={onDownloadSketchClick}
							className="cursor-pointer flex items-center justify-center gap-2 rounded-b-xl bg-white px-4 py-4 text-sm font-bold text-neutral-900 transition-all hover:bg-neutral-200 dark:bg-neutral-900 dark:text-neutral-100 dark:hover:bg-neutral-800 border-x-2 border-b-2 border-neutral-200 dark:border-neutral-800">
							<FaSave size={16} />
						</button>
					</div>
				</div>

				<div className="flex flex-row rounded-xl border-2 border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900 p-4 w-full max-w-md gap-4">
					<div className="flex items-center gap-2">
						<label
							className="h-12 w-12 rounded-lg cursor-pointer overflow-hidden transition-all duration-300 hover:scale-105"
							style={{ backgroundColor: color }}
							title="Stroke Color">
							<input
								type="color"
								value={color}
								onChange={e => setColor(e.target.value)}
								className="opacity-0 w-full h-full cursor-pointer"
								aria-label="Stroke Color"
							/>
						</label>
					</div>

					<div className="flex flex-col w-full">
						<div className="flex items-center justify-between">
							<p className="text-sm font-semibold text-neutral-800 dark:text-neutral-100">
								{drawAction === "eraser" ? "Eraser Size" : "Stroke Size"}
							</p>
							<p className="text-sm text-neutral-600 dark:text-neutral-300">
								{strokeWidth}px
							</p>
						</div>

						<input
							type="range"
							min={1}
							max={40}
							value={strokeWidth}
							onChange={e => setStrokeWidth(Number(e.target.value))}
							className="mt-2 w-full accent-blue-600"
						/>

						<div className="mt-1 flex justify-between text-xs text-neutral-500 dark:text-neutral-400">
							<span>Thin</span>
							<span>Thick</span>
						</div>
					</div>
				</div>

				<button
					type="button"
					onClick={async () => {
						const blob = await exportSketchBlob();
						convertToGraphviz(blob);
					}}
					className="w-full max-w-md cursor-pointer flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-4 text-sm font-bold text-neutral-100 transition-all hover:bg-blue-700">
					<FaCode size={16} />
					Convert to Graphviz
				</button>
			</div>
		</div>
	);
};
