import React, { useCallback, useRef, useState } from "react";
import {
	ArrowUpRight,
	Circle,
	Code,
	Download,
	Edit3,
	MousePointer,
	Square,
	Trash2,
	Upload,
} from "react-feather";
import {
	Arrow as KonvaArrow,
	Circle as KonvaCircle,
	Image as KonvaImage,
	Line as KonvaLine,
	Rect as KonvaRect,
	Layer,
	Stage,
	Transformer,
} from "react-konva";
import { v4 as uuidv4 } from "uuid";

const DrawAction = {
	Select: "select",
	Rectangle: "rectangle",
	Circle: "circle",
	Pen: "freedraw",
	Arrow: "arrow",
};

const PaintOptions = [
	{
		id: DrawAction.Select,
		label: "Select",
		icon: <MousePointer size={16} />,
	},
	{
		id: DrawAction.Rectangle,
		label: "Rectangle",
		icon: <Square size={16} />,
	},
	{
		id: DrawAction.Circle,
		label: "Circle",
		icon: <Circle size={16} />,
	},
	{
		id: DrawAction.Arrow,
		label: "Arrow",
		icon: <ArrowUpRight size={16} />,
	},
	{
		id: DrawAction.Pen,
		label: "Pen",
		icon: <Edit3 size={16} />,
	},
];

const downloadURI = (uri, name) => {
	const link = document.createElement("a");
	link.download = name;
	link.href = uri || "";
	document.body.appendChild(link);
	link.click();
	document.body.removeChild(link);
};

export const GraphSketchpad = () => {
	const [color, setColor] = useState("#000000");
	const [drawAction, setDrawAction] = useState(DrawAction.Pen);
	const [canvasSize, setCanvasSize] = useState(768);

	const [pens, setPens] = useState([]);
	const [rectangles, setRectangles] = useState([]);
	const [circles, setCircles] = useState([]);
	const [arrows, setArrows] = useState([]);
	const [image, setImage] = useState(undefined);

	const fileRef = useRef(null);
	const stageRef = useRef(null);
	const transformerRef = useRef(null);

	const onImportImageSelect = useCallback(e => {
		const file = e.target.files?.[0];
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

	const onDownloadSketchClick = useCallback(() => {
		if (!stageRef.current) return;

		// Export target size (fixed)
		const exportSize = 768;
		// displayed canvas size (fallback if canvasSize is falsy)
		const displayedSize =
			canvasSize ||
			Math.max(stageRef.current.width(), stageRef.current.height()) ||
			exportSize;

		// pixelRatio scales the current canvas to the target export size
		const pixelRatio = displayedSize > 0 ? exportSize / displayedSize : 1;

		const dataUri = stageRef.current.toDataURL({
			pixelRatio,
			mimeType: "image/png",
		});
		downloadURI(dataUri, "sketch.png");
	}, [canvasSize]);

	const onClear = useCallback(() => {
		setRectangles([]);
		setCircles([]);
		setPens([]);
		setArrows([]);
		setImage(undefined);

		if (transformerRef.current) {
			try {
				transformerRef.current.nodes([]);
				if (transformerRef.current.getLayer)
					transformerRef.current.getLayer().batchDraw();
			} catch {
				// ignore if transformer is not ready
			}
		}
	}, []);

	const isPaintRef = useRef(false);
	const currentShapeRef = useRef(undefined);

	const onStageMouseUp = useCallback(() => {
		isPaintRef.current = false;
	}, []);

	const onStageMouseDown = useCallback(() => {
		if (drawAction === DrawAction.Select) return;

		isPaintRef.current = true;

		const stage = stageRef.current;
		const pos = stage.getPointerPosition();

		const x = pos.x;
		const y = pos.y;

		const id = uuidv4();
		currentShapeRef.current = id;

		if (drawAction === DrawAction.Pen) {
			setPens(prev => [...prev, { id, points: [x, y], color }]);
		} else if (drawAction === DrawAction.Circle) {
			setCircles(prev => [...prev, { id, radius: 1, x, y, color }]);
		} else if (drawAction === DrawAction.Rectangle) {
			setRectangles(prev => [
				...prev,
				{ id, height: 1, width: 1, x, y, color },
			]);
		} else if (drawAction === DrawAction.Arrow) {
			setArrows(prev => [...prev, { id, points: [x, y, x, y], color }]);
		}
	}, [drawAction, color]);

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
		e => {
			if (drawAction !== DrawAction.Select) return;
			const currentTarget = e.currentTarget;

			if (!transformerRef.current) return;
			transformerRef.current.nodes([currentTarget]);
			if (transformerRef.current.getLayer)
				transformerRef.current.getLayer().batchDraw();
		},
		[drawAction]
	);

	const onBgClick = useCallback(() => {
		if (!transformerRef.current) return;
		try {
			transformerRef.current.nodes([]);
			if (transformerRef.current.getLayer)
				transformerRef.current.getLayer().batchDraw();
		} catch {
			// ignore
		}
	}, []);

	const isDraggable = drawAction === DrawAction.Select;

	// compute image display sizing to preserve aspect ratio and center inside canvas
	let imageRenderProps = null;
	if (image) {
		const imgW = image.naturalWidth || image.width || 1;
		const imgH = image.naturalHeight || image.height || 1;
		const canvasW = canvasSize;
		const canvasH = canvasSize;

		const scale = Math.min(canvasW / imgW, canvasH / imgH);
		const displayW = Math.round(imgW * scale);
		const displayH = Math.round(imgH * scale);
		const offsetX = Math.round((canvasW - displayW) / 2);
		const offsetY = Math.round((canvasH - displayH) / 2);

		imageRenderProps = {
			x: offsetX,
			y: offsetY,
			width: displayW,
			height: displayH,
		};
	}

	return (
		<div className="w-full md:w-1/3 flex flex-col min-h-0">
			<h2 className="mb-4 text-xl font-semibold">Sketch your Graph</h2>
			<div
				className="w-full overflow-hidden rounded-xl border-2 border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 shadow-md"
				style={{ aspectRatio: "1 / 1" }}>
				<Stage
					width={canvasSize}
					height={canvasSize}
					ref={stageRef}
					onMouseUp={onStageMouseUp}
					onMouseDown={onStageMouseDown}
					onMouseMove={onStageMouseMove}
					style={{ margin: "0 auto", display: "block" }}>
					<Layer>
						<KonvaRect
							x={0}
							y={0}
							height={canvasSize}
							width={canvasSize}
							fill="white"
							id="bg"
							onClick={onBgClick}
						/>

						{image && (
							<KonvaImage
								image={image}
								x={imageRenderProps.x}
								y={imageRenderProps.y}
								width={imageRenderProps.width}
								height={imageRenderProps.height}
								draggable={isDraggable}
								onClick={onShapeClick}
							/>
						)}

						{arrows.map(arrow => (
							<KonvaArrow
								key={arrow.id}
								id={arrow.id}
								points={arrow.points}
								fill={arrow.color}
								stroke={arrow.color}
								strokeWidth={4}
								onClick={onShapeClick}
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
								strokeWidth={4}
								onClick={onShapeClick}
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
								strokeWidth={4}
								onClick={onShapeClick}
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
								strokeWidth={4}
								points={pen.points}
								onClick={onShapeClick}
								draggable={isDraggable}
							/>
						))}

						<Transformer ref={transformerRef} />
					</Layer>
				</Stage>
			</div>
			<div className="flex flex-col gap-2 mt-4 items-center">
				<div className="flex gap-4">
					<div className="flex items-center overflow-hidden rounded-xl border-2 border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900 w-fit">
						{PaintOptions.map(({ id, label, icon }) => {
							const active = id === drawAction;
							return (
								<button
									key={id}
									type="button"
									title={label}
									onClick={() => setDrawAction(id)}
									className={`flex items-center gap-2 px-4 py-4 text-md font-semibold transition-all duration-300 rounded-xl cursor-pointer ${
										active
											? "bg-blue-600 text-white"
											: "bg-white text-neutral-900 hover:bg-neutral-200 dark:bg-neutral-900 dark:text-neutral-100 dark:hover:bg-neutral-800"
									}`}>
									{icon}
									{/* <span className="hidden sm:inline">{label}</span> */}
								</button>
							);
						})}

						<div className="flex items-center gap-2 px-3 py-2">
							<label
								className="h-8 w-8 rounded-lg cursor-pointer overflow-hidden transition-all duration-300"
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
							{/* <span className="hidden text-md text-neutral-900 dark:text-neutral-100 sm:inline">
							{color}
						</span> */}
						</div>

						<button
							type="button"
							onClick={onClear}
							title="Clear"
							className="flex items-center gap-2 px-4 py-4 text-md font-semibold transition-all duration-300 rounded-xl cursor-pointer bg-white text-neutral-900 hover:bg-neutral-200 dark:bg-neutral-900 dark:text-neutral-100 dark:hover:bg-neutral-800">
							<Trash2 size={16} />
							{/* <span className="hidden sm:inline">Clear</span> */}
						</button>
					</div>
					<div className="flex items-center gap-1">
						<input
							type="file"
							ref={fileRef}
							onChange={onImportImageSelect}
							style={{ display: "none" }}
							accept="image/*"
						/>

						<button
							type="button"
							onClick={onImportImageClick}
							className="cursor-pointer flex items-center gap-2 rounded-xl bg-white px-4 py-4 text-sm font-bold text-neutral-900 transition-all hover:bg-neutral-200 dark:bg-neutral-900 dark:text-neutral-100 dark:hover:bg-neutral-800 border-2 border-neutral-200 dark:border-neutral-800">
							<Upload size={16} />
							{/* Import Image */}
						</button>

						<button
							type="button"
							onClick={onDownloadSketchClick}
							className="cursor-pointer flex items-center gap-2 rounded-xl bg-white px-4 py-4 text-sm font-bold text-neutral-900 transition-all hover:bg-neutral-200 dark:bg-neutral-900 dark:text-neutral-100 dark:hover:bg-neutral-800 border-2 border-neutral-200 dark:border-neutral-800">
							<Download size={16} />
							{/* Download Sketch */}
						</button>
					</div>
				</div>

				<button
					type="button"
					onClick={onDownloadSketchClick}
					className="cursor-pointer flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-4 text-sm font-bold text-neutral-100 transition-all hover:bg-blue-700 border-2 border-neutral-200 dark:border-neutral-800">
					<Code size={16} />
					Convert to Graphviz
				</button>
			</div>
		</div>
	);
};
