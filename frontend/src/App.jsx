import './App.css';
import ModelViewer from './components/ModelViewer';
import React, { useEffect, useRef, useState } from 'react';
import Toolbar from './components/Toolbar';
import ImproveDialog from './components/ImproveDialog';

import {
	ContextMenu,
	ContextMenuContent,
	ContextMenuItem,
	ContextMenuTrigger,
} from '@/components/ui/context-menu';

import {
	Dialog,
	DialogClose,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from '@/components/ui/dialog';

const DrawingApp = () => {
	const canvasRef = useRef(null);
	const ctxRef = useRef(null);
	const isPaintingRef = useRef(false);

	const [lineWidth, setLineWidth] = useState(5);
	const [strokeColor, setStrokeColor] = useState('#000000');
	const [isEraser, setIsEraser] = useState(false);
	const [undoStack, setUndoStack] = useState([]);
	const [showModel, setShowModel] = useState(false);

	const lineWidthRef = useRef(lineWidth);
	const strokeColorRef = useRef(strokeColor);
	const isEraserRef = useRef(isEraser);
	const undoStackRef = useRef(undoStack);

	// Keep refs in sync with state
	useEffect(() => {
		lineWidthRef.current = lineWidth;
	}, [lineWidth]);

	useEffect(() => {
		strokeColorRef.current = strokeColor;
	}, [strokeColor]);

	useEffect(() => {
		isEraserRef.current = isEraser;
	}, [isEraser]);

	useEffect(() => {
		undoStackRef.current = undoStack;
	}, [undoStack]);

	const handleMouseDown = (e) => {
		const canvas = canvasRef.current;
		const ctx = ctxRef.current;

		isPaintingRef.current = true;
		ctx.beginPath();
		ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
		setUndoStack((prev) => [
			...prev,
			ctx.getImageData(0, 0, canvas.width, canvas.height),
		]);
	};

	const handleMouseUp = () => {
		isPaintingRef.current = false;
		const ctx = ctxRef.current;
		ctx.stroke();
		ctx.beginPath();
	};

	const handleMouseMove = (e) => {
		if (!isPaintingRef.current) return;
		const canvas = canvasRef.current;
		const ctx = ctxRef.current;

		ctx.lineWidth = lineWidthRef.current;
		ctx.lineCap = 'round';

		if (isEraserRef.current) {
			ctx.strokeStyle = 'white';
			const eraserRadius = 20;
			ctx.beginPath();
			ctx.arc(
				e.clientX - canvas.offsetLeft,
				e.clientY - canvas.offsetTop,
				eraserRadius,
				0,
				2 * Math.PI
			);
			ctx.fillStyle = 'white';
			ctx.fill();
			ctx.closePath();
		} else {
			// Only draw when right mouse button is pressed (button 2)
			if (e.buttons === 1) {
				ctx.strokeStyle = strokeColorRef.current;
				ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
				ctx.stroke();
			} else {
				// If not right click, just move without drawing
				ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
				return;
			}
		}
	};

	const handleKeyDown = (e) => {
		const stack = undoStackRef.current;
		if (e.ctrlKey && e.key.toLowerCase() === 'z') {
			console.log('undo stack: ' + stack.length);
			e.preventDefault();
			if (stack.length > 0) {
				const ctx = ctxRef.current;
				const newStack = [...stack];
				const last = newStack.pop();
				ctx.putImageData(last, 0, 0);
				setUndoStack(newStack);
			}
		}
	};

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;

		const ctx = canvas.getContext('2d');
		ctxRef.current = ctx;

		canvas.width = window.innerWidth - canvas.offsetLeft;
		canvas.height = window.innerHeight - canvas.offsetTop;

		ctx.fillStyle = 'white';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		// Save initial state to undo stack
		setUndoStack([ctx.getImageData(0, 0, canvas.width, canvas.height)]);

		canvas.addEventListener('mousedown', handleMouseDown);
		canvas.addEventListener('mouseup', handleMouseUp);
		canvas.addEventListener('mousemove', handleMouseMove);
		document.addEventListener('keydown', handleKeyDown);

		return () => {
			canvas.removeEventListener('mousedown', handleMouseDown);
			canvas.removeEventListener('mouseup', handleMouseUp);
			canvas.removeEventListener('mousemove', handleMouseMove);
			document.removeEventListener('keydown', handleKeyDown);
		};
	}, [showModel]); //

	const handleClear = () => {
		const canvas = canvasRef.current;
		const ctx = ctxRef.current;
		const goodImg = ctx.getImageData(0, 0, canvas.width, canvas.height);
		let newStack = [undoStackRef.current, goodImg];
		setUndoStack(newStack);
		console.log(newStack.length);
		ctx.clearRect(0, 0, canvas.width, canvas.height);

		ctx.fillStyle = 'white';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
	};

	const handleUndo = () => {
		console.log('stack:' + undoStack.length);
		const ctx = ctxRef.current;
		if (undoStack.length > 0) {
			const newStack = [...undoStack];
			const last = newStack.pop();
			ctx.putImageData(last, 0, 0);
			setUndoStack(newStack);
		}
	};

	const [exportedImage, setExportedImage] = useState(null);

	const handleDownload = () => {
		const canvas = canvasRef.current;
		const ctx = canvas.getContext('2d');
		const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
		const data = imageData.data;

		// Find the boundaries of the drawing (non-white pixels)
		let minX = canvas.width,
			minY = canvas.height,
			maxX = 0,
			maxY = 0;
		let hasDrawing = false;

		for (let y = 0; y < canvas.height; y++) {
			for (let x = 0; x < canvas.width; x++) {
				const idx = (y * canvas.width + x) * 4;
				// Check if pixel is not white (RGB not all 255)
				if (
					data[idx] !== 255 ||
					data[idx + 1] !== 255 ||
					data[idx + 2] !== 255
				) {
					minX = Math.min(minX, x);
					minY = Math.min(minY, y);
					maxX = Math.max(maxX, x);
					maxY = Math.max(maxY, y);
					hasDrawing = true;
				}
			}
		}

		// If no drawing found, return the entire canvas
		if (!hasDrawing) {
			setExportedImage(canvas.toDataURL('image/png'));
			return;
		}

		// Add some padding
		const padding = 10;
		minX = Math.max(0, minX - padding);
		minY = Math.max(0, minY - padding);
		maxX = Math.min(canvas.width, maxX + padding);
		maxY = Math.min(canvas.height, maxY + padding);

		// Create a new canvas with just the drawing area
		const croppedWidth = maxX - minX;
		const croppedHeight = maxY - minY;

		// Only proceed if we have a valid crop area
		if (croppedWidth > 0 && croppedHeight > 0) {
			const tempCanvas = document.createElement('canvas');
			tempCanvas.width = croppedWidth;
			tempCanvas.height = croppedHeight;
			const tempCtx = tempCanvas.getContext('2d');

			// Draw the cropped region to the temp canvas
			tempCtx.drawImage(
				canvas,
				minX,
				minY,
				croppedWidth,
				croppedHeight,
				0,
				0,
				croppedWidth,
				croppedHeight
			);

			// Convert to data URL
			setExportedImage(tempCanvas.toDataURL('image/png'));
		} else {
			// Fallback to full canvas if something went wrong
			setExportedImage(canvas.toDataURL('image/png'));
		}
	};

	const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });

	// Add mouse move handler for cursor
	useEffect(() => {
		const handleCursorMove = (e) => {
			setCursorPos({ x: e.clientX, y: e.clientY });
		};

		document.addEventListener('mousemove', handleCursorMove);
		return () => {
			document.removeEventListener('mousemove', handleCursorMove);
		};
	}, []);

	const [genAImodel, setGenAImodel] = useState(null);
	const [open, setOpen] = useState(false);
	return (
		<>
			<Dialog open={open} onOpenChange={setOpen}>
				<ImproveDialog
					imageUrl={exportedImage}
					onOpenChange={setOpen}
					toggleModelShow={setShowModel}
					setModelUrl={setGenAImodel}
				/>

				<ContextMenu>
					<ContextMenuContent className="dark w-48">
						<DialogTrigger asChild>
							<ContextMenuItem inset onClick={() => handleDownload()}>
								Improve with AI
							</ContextMenuItem>
						</DialogTrigger>

						<ContextMenuItem inset>Clear Canvas</ContextMenuItem>
						<ContextMenuItem inset>Undo Previous</ContextMenuItem>
					</ContextMenuContent>
					<div
						className={`pointer-events-none w-[40px] h-[40px] border-neutral-900 border-2 rounded-full fixed`}
						style={{
							left: `${cursorPos.x - 20}px`,
							top: `${cursorPos.y - 20}px`,
							zIndex: 9999,
							visibility: isEraser ? 'visible' : 'hidden',
						}}
					></div>
					<section className="w-screen h-screen flex">
						<Toolbar
							strokeColor={strokeColor}
							setStrokeColor={setStrokeColor}
							lineWidth={lineWidth}
							setLineWidth={setLineWidth}
							isEraser={isEraser}
							setIsEraser={setIsEraser}
							handleClear={handleClear}
							handleUndo={handleUndo}
							handleDownload={handleDownload}
							showModel={showModel}
							setShowModel={setShowModel}
							modelUrl={genAImodel}
						/>
						{showModel ? (
							<ModelViewer modelUrl={genAImodel} />
						) : (
							<ContextMenuTrigger>
								<div className="drawing-board grow flex">
									<canvas
										ref={canvasRef}
										id="drawing-board"
										className="flex-grow"
									/>
								</div>
							</ContextMenuTrigger>
						)}
					</section>
				</ContextMenu>
			</Dialog>
		</>
	);
};

export default DrawingApp;
