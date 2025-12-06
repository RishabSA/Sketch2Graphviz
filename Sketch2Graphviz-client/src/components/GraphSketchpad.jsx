import React, { useEffect, useRef, useState } from "react";
import { FileText, Upload } from "react-feather";

export const GraphSketchpad = ({ onConvert }) => {
	const canvasRef = useRef(null);
	const fileInputRef = useRef(null);
	const [isDrawing, setIsDrawing] = useState(false);
	const [lastPoint, setLastPoint] = useState(null);

	// Resize canvas to fit container
	const containerRef = useRef(null);

	useEffect(() => {
		const resize = () => {
			const canvas = canvasRef.current;
			const container = containerRef.current;
			if (!canvas || !container) return;

			const rect = container.getBoundingClientRect();
			const ctx = canvas.getContext("2d");
			if (!ctx) return;

			canvas.width = rect.width;
			canvas.height = rect.height;

			// Fill background white
			ctx.fillStyle = "#ffffff";
			ctx.fillRect(0, 0, canvas.width, canvas.height);
		};

		resize();
		window.addEventListener("resize", resize);
		return () => window.removeEventListener("resize", resize);
	}, []);

	const getCanvasPos = e => {
		const canvas = canvasRef.current;
		if (!canvas) return { x: 0, y: 0 };
		const rect = canvas.getBoundingClientRect();
		return {
			x: e.clientX - rect.left,
			y: e.clientY - rect.top,
		};
	};

	const handleMouseDown = e => {
		const pos = getCanvasPos(e);
		setIsDrawing(true);
		setLastPoint(pos);
	};

	const handleMouseMove = e => {
		if (!isDrawing || !lastPoint) return;
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		const pos = getCanvasPos(e);
		ctx.strokeStyle = "#000000";
		ctx.lineWidth = 2;
		ctx.lineCap = "round";

		ctx.beginPath();
		ctx.moveTo(lastPoint.x, lastPoint.y);
		ctx.lineTo(pos.x, pos.y);
		ctx.stroke();

		setLastPoint(pos);
	};

	const stopDrawing = () => {
		setIsDrawing(false);
		setLastPoint(null);
	};

	const handleUploadClick = () => {
		fileInputRef.current?.click();
	};

	const handleFileChange = e => {
		const file = e.target.files?.[0];
		if (!file) return;

		const reader = new FileReader();
		reader.onload = () => {
			const img = new Image();
			img.onload = () => {
				const canvas = canvasRef.current;
				if (!canvas) return;
				const ctx = canvas.getContext("2d");
				if (!ctx) return;

				// Clear and draw white background
				ctx.fillStyle = "#ffffff";
				ctx.fillRect(0, 0, canvas.width, canvas.height);

				// Fit image into canvas and preserve the aspect ratio
				const canvasRatio = canvas.width / canvas.height;
				const imgRatio = img.width / img.height;

				let drawWidth = canvas.width;
				let drawHeight = canvas.height;
				if (imgRatio > canvasRatio) {
					// Image wider than canvas
					drawWidth = canvas.width;
					drawHeight = canvas.width / imgRatio;
				} else {
					// Image taller than canvas
					drawHeight = canvas.height;
					drawWidth = canvas.height * imgRatio;
				}

				const offsetX = (canvas.width - drawWidth) / 2;
				const offsetY = (canvas.height - drawHeight) / 2;

				ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
			};
			if (typeof reader.result === "string") {
				img.src = reader.result;
			}
		};
		reader.readAsDataURL(file);
	};

	const handleConvertClick = () => {
		if (onConvert) {
			onConvert(canvasRef.current);
		} else {
			console.log("Convert to Graphviz Code (wire this to your backend)");
		}
	};

	return (
		<>
			<input
				ref={fileInputRef}
				type="file"
				accept="image/*"
				className="hidden"
				onChange={handleFileChange}
			/>

			<button
				onClick={handleUploadClick}
				className="mb-2 flex items-center gap-2 w-full cursor-pointer text-white bg-blue-600 hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-700 focus:outline-none font-semibold rounded-lg text-sm px-4 py-3">
				<Upload size={24} className="stroke-current text-white" />
				Upload Image
			</button>

			<button
				onClick={handleConvertClick}
				className="mb-3 flex items-center gap-2 w-full cursor-pointer text-white bg-blue-600 hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-700 focus:outline-none font-semibold rounded-lg text-sm px-4 py-3">
				<FileText size={24} className="stroke-current text-white" />
				Convert to Graphviz Code
			</button>

			<div
				ref={containerRef}
				className="flex-1 bg-neutral-100 dark:bg-neutral-900 border-neutral-200 dark:border-neutral-800 border-2 rounded-xl shadow-md overflow-hidden">
				<canvas
					ref={canvasRef}
					className="w-full h-full cursor-crosshair"
					onMouseDown={handleMouseDown}
					onMouseMove={handleMouseMove}
					onMouseUp={stopDrawing}
					onMouseLeave={stopDrawing}
				/>
			</div>
		</>
	);
};
