import { useState, useEffect } from 'react';
import {
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
	DialogFooter,
	DialogClose,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';

const ImproveDialog = ({
	onOpenChange,
	imageUrl,
	toggleModelShow,
	setModelUrl,
}) => {
	const [prompt, setPrompt] = useState('');
	const [negativePrompt, setNegativePrompt] = useState('');
	const [isSubmitting, setIsSubmitting] = useState(false);
	const [isGeneratingModel, setIsGeneratingModel] = useState(false);

	const [aiUrl, setAiURL] = useState(null);

	const [progress, setProgress] = useState(0);
	useEffect(() => {
		if (isSubmitting) {
			let value = 0;
			const interval = setInterval(() => {
				value = Math.min(value + Math.random() * 10, 95); // simulate until 95%
				setProgress(value);
			}, 300);

			return () => clearInterval(interval);
		}
	}, [isSubmitting]);

	const handleSubmit = async () => {
		if (!prompt.trim()) return;
		setProgress(0);
		setIsSubmitting(true);
		try {
			// Convert image URL to binary
			const response = await fetch(imageUrl);
			const blob = await response.blob();

			// Create FormData to send to backend
			const formData = new FormData();
			formData.append('prompt', prompt);
			formData.append('sketch', blob, 'sketch.png');

			if (negativePrompt.trim()) {
				formData.append('negative_prompt', negativePrompt);
			}

			// Send to backend

			const result = await fetch('http://localhost:8000/generate-image', {
				method: 'POST',
				body: formData,
			});

			if (!result.ok) {
				throw new Error(`HTTP error! status: ${result.status}`);
			}

			const resultBlob = await result.blob();
			const temp_aiUrl = URL.createObjectURL(resultBlob);

			setAiURL(temp_aiUrl);

			setProgress(100);
			setIsSubmitting(false);
			// Process response
			// possibly open a secondary dialog here:
		} catch (error) {
			console.error('Error improving image:', error);
			setIsSubmitting(false);
		}
	};

	const handleGenerate = async () => {
		if (!aiUrl) return;

		try {
			// Fetch the AI generated image

			setIsGeneratingModel(true);
			const response = await fetch(aiUrl);
			const blob = await response.blob();

			// Create FormData to send to backend
			const formData = new FormData();
			formData.append('image', blob, 'ai_image.png');

			// Send to backend for 3D model generation
			const result = await fetch('http://localhost:8000/generate-model', {
				method: 'POST',
				body: formData,
			});

			if (!result.ok) {
				throw new Error(`HTTP error! status: ${result.status}`);
			}

			// Handle the GLB file returned from the backend
			const modelBlob = await result.blob();
			const modelUrl = URL.createObjectURL(modelBlob);

			// Instead of downloading, store the model URL in global state
			// Using a custom event to communicate with parent components
			setModelUrl(modelUrl);
			setIsGeneratingModel(false);
			// Close the dialog and signal to toggle the model viewer
			onOpenChange(false);
			toggleModelShow(true);
		} catch (error) {
			console.error('Error generating 3D model:', error);
		}
	};

	const renderStatusMessage = (progress) => {
		if (progress < 15) return 'Sampling run 1...';
		if (progress < 30) return 'Sampling run 2...';
		if (progress < 45) return 'Generating mesh...';
		if (progress < 60) return 'Decimating mesh...';
		if (progress < 75) return 'Removing extra faces and vertices...';
		if (progress < 90) return 'Baking in textures...';
		return 'Optimizing textures...';
	};

	useEffect(() => {
		if (isGeneratingModel) {
			let current = 0;
			const interval = setInterval(() => {
				current = Math.min(current + Math.random() * 5, 99);
				setProgress(current);
			}, 300);
			return () => clearInterval(interval);
		}
	}, [isGeneratingModel]);

	return (
		<DialogContent className="sm:max-w-[750px] dark">
			<DialogHeader>
				<DialogTitle className="text-xl font-semibold">
					Enhance Drawing
				</DialogTitle>
				<DialogDescription>
					Enhance your sketch using AI. Provide prompts to guide the
					transformation.
				</DialogDescription>
			</DialogHeader>

			<div className="grid gap-4 py-4">
				<div className="flex items-center justify-center mb-2 gap-4">
					{/* Original Sketch */}
					<div className="relative w-full max-w-[260px] aspect-[4/3] rounded-md border border-border shadow-sm bg-muted/30 overflow-hidden flex items-center justify-center">
						{imageUrl ? (
							<img
								src={imageUrl}
								alt="Current drawing"
								className="w-full h-full object-contain"
							/>
						) : (
							<span className="text-muted-foreground">No image available</span>
						)}
					</div>

					{/* AI Generated */}
					<div className="relative w-full max-w-[260px] aspect-[4/3] rounded-md border border-border shadow-sm bg-muted/30 overflow-hidden flex items-center justify-center">
						{aiUrl ? (
							<img
								src={aiUrl}
								alt="AI generated result"
								className="w-full h-full object-contain"
							/>
						) : isSubmitting ? (
							<div className="flex flex-col items-center justify-center w-full px-4">
								<Progress value={progress} className="w-full h-2" />
								<p className="text-xs text-muted-foreground mt-2">
									Generating image...
								</p>
							</div>
						) : (
							<span className="text-muted-foreground">
								Nothing generated yet!
							</span>
						)}
					</div>
				</div>

				<Tabs defaultValue="prompt" className="w-full">
					<TabsList className="grid w-full grid-cols-2">
						<TabsTrigger value="prompt">Prompt</TabsTrigger>
						<TabsTrigger value="advanced">Advanced</TabsTrigger>
					</TabsList>
					<TabsContent value="prompt" className="space-y-4 mt-2">
						<div className="space-y-2">
							<Label htmlFor="prompt">Describe what you want to create</Label>
							<Textarea
								id="prompt"
								placeholder="A beautiful landscape with mountains and lakes..."
								value={prompt}
								onChange={(e) => setPrompt(e.target.value)}
								className="min-h-[100px]"
							/>
						</div>
					</TabsContent>

					<TabsContent value="advanced" className="space-y-4 mt-2">
						<div className="space-y-2">
							<Label htmlFor="prompt">Prompt</Label>
							<Textarea
								id="prompt"
								placeholder="A beautiful landscape with mountains and lakes..."
								value={prompt}
								onChange={(e) => setPrompt(e.target.value)}
								className="min-h-[80px]"
							/>
						</div>

						<Separator className="my-2" />

						<div className="space-y-2">
							<Label htmlFor="negative-prompt">
								Negative Prompt (Optional)
							</Label>
							<Textarea
								id="negative-prompt"
								placeholder="Elements to avoid: blurry, distorted, low quality..."
								value={negativePrompt}
								onChange={(e) => setNegativePrompt(e.target.value)}
								className="min-h-[80px]"
							/>
							<p className="text-xs text-muted-foreground">
								Specify elements you want to avoid in the generated image
							</p>
						</div>
					</TabsContent>
				</Tabs>
			</div>

			<DialogFooter>
				<DialogClose asChild>
					<Button variant="outline" onClick={() => onOpenChange(false)}>
						Cancel
					</Button>
				</DialogClose>
				<Button
					onClick={handleSubmit}
					disabled={!prompt.trim() || isSubmitting || !imageUrl}
					className="ml-2"
				>
					{isSubmitting ? 'Processing...' : 'Enhance'}
				</Button>
				<Button
					onClick={handleGenerate}
					disabled={!aiUrl || isGeneratingModel}
					className={aiUrl ? 'glow-button' : ''}
				>
					Model
				</Button>
			</DialogFooter>
			{isGeneratingModel && (
				<div className="absolute inset-0 z-50 bg-black/40 backdrop-blur-md flex flex-col items-center justify-center rounded-md px-4">
					<div className="w-full max-w-xs">
						<Progress value={progress} className="h-2 w-full mb-4" />
					</div>
					<p className="text-white text-sm text-center">
						{renderStatusMessage(progress)}
					</p>
				</div>
			)}
		</DialogContent>
	);
};

export default ImproveDialog;
