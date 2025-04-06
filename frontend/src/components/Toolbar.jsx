import { Button } from '@/components/ui/button';
import { DialogTrigger } from '@/components/ui/dialog';

import ColorPickerInput from './ui/colorpicker';
const Toolbar = ({
	strokeColor,
	setStrokeColor,
	lineWidth,
	setLineWidth,
	isEraser,
	setIsEraser,
	handleClear,
	handleUndo,
	handleDownload,
	showModel,
	setShowModel,
	modelUrl,
}) => {
	return (
		<div className="bg-neutral-800 p-6 shadow-md space-y-6 w-[350px]">
			<h1 className="text-2xl font-bold text-gray-100 w-[350px]">
				hAIgher dimension
			</h1>

			<div className="flex flex-wrap gap-6 flex-col">
				<div className="form-control space-y-2">
					<label
						htmlFor="stroke"
						className="block text-sm font-medium text-gray-50"
					>
						Stroke
					</label>
					<ColorPickerInput
						id="stroke"
						value={strokeColor}
						onChange={(e) => setStrokeColor(e.target.value)}
						className="w-16 h-8 border border-gray-300 rounded-md"
					/>
				</div>

				<div className="form-control space-y-2">
					<label
						htmlFor="lineWidth"
						className="block text-sm font-medium text-gray-50"
					>
						Line Width
					</label>
					<input
						id="lineWidth"
						type="number"
						min="1"
						max="100"
						value={lineWidth}
						onChange={(e) => setLineWidth(Number(e.target.value))}
						className="w-full p-2 border border-gray-300 rounded-md"
					/>
					<div className="flex space-x-2 mt-2 flex-col">
						<div>
							{[2, 4, 6, 8, 10].map((width) => (
								<Button
									key={width}
									onClick={() => setLineWidth(width)}
									variant={lineWidth === width ? 'default' : 'secondary'}
									className={`m-2 w-[40px] h-[40px] p-0 ${
										lineWidth === width
											? 'bg-blue-500'
											: 'bg-gray-300 text-gray-800'
									}`}
								>
									{width}
								</Button>
							))}
						</div>
						<div>
							{[12, 14, 16, 18, 20].map((width) => (
								<Button
									key={width}
									onClick={() => setLineWidth(width)}
									variant={lineWidth === width ? 'default' : 'secondary'}
									className={`m-2 w-[40px] h-[40px] p-0 ${
										lineWidth === width
											? 'bg-blue-500'
											: 'bg-gray-300 text-gray-800'
									}`}
								>
									{width}
								</Button>
							))}
						</div>
					</div>
				</div>

				<div className="form-control flex items-center space-x-2">
					<label className="text-sm font-medium text-gray-50">Mode</label>
					<div className="flex space-x-2">
						<Button
							onClick={() => setIsEraser(false)}
							variant={!isEraser ? 'default' : 'secondary'}
							className={
								!isEraser ? 'bg-blue-500' : 'bg-gray-300 text-gray-800'
							}
						>
							Brush
						</Button>
						<Button
							onClick={() => setIsEraser(true)}
							variant={isEraser ? 'default' : 'secondary'}
							className={isEraser ? 'bg-blue-500' : 'bg-gray-300 text-gray-800'}
						>
							Eraser
						</Button>
					</div>
				</div>

				<div className="space-y-2">
					<Button
						onClick={handleClear}
						variant="destructive"
						className="w-full bg-red-800 hover:bg-red-700"
					>
						Clear
					</Button>
					<Button
						onClick={handleUndo}
						variant="default"
						className="w-full bg-yellow-800 hover:bg-yellow-700"
					>
						Undo
					</Button>
					<DialogTrigger asChild>
						<Button
							variant="default"
							className="w-full bg-green-800 hover:bg-green-700"
						>
							Improve with AI
						</Button>
					</DialogTrigger>
					<Button
						onClick={() => setShowModel(!showModel)}
						variant="default"
						className="w-full bg-blue-800 hover:bg-blue-700"
						disabled={!modelUrl}
					>
						Toggle Model
					</Button>
				</div>
			</div>
		</div>
	);
};

export default Toolbar;
