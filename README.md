## Inspiration

We’ve all had that moment where we doodle something cool on paper or a tablet and think, “This could totally be a 3D model.” But for most people, turning a sketch into a 3D object feels like a long, technical, often intimidating process. We wanted to simplify that. Lazy3D was born out of a desire to make 3D creation as easy as sketching on a napkin — something casual, creative, and fast.

## What it does

Lazy3D takes a user’s colorful sketch and a short prompt, then automatically turns it into a full 3D model in minutes. It first uses a generative AI model to convert the sketch into a clean, high-quality image, and then passes that image to a 3D generation tool to produce a downloadable 3D file. The whole process happens in just a couple of clicks — no modeling skills needed.

## How we built it

We built a web app with a JavaScript frontend that lets users upload or draw sketches and enter prompts. That data is sent to our Python backend, where ControlNet (Scribble model) transforms the sketch into a vivid, Trellis-friendly image. We then pass that image to Trellis, which generates the final 3D model. The image and model are streamed back to the frontend, where the user can view and interact with them.

## Challenges we ran into

- Getting ControlNet to generate consistent, clean images from loose, colorful sketches took a lot of prompt tuning and postprocessing.
- Making sure the resulting image worked well with Trellis involved experimenting with resolution, transparency, and layout.
- Integrating everything into a smooth frontend-backend pipeline (especially large image payloads) took some time to get right.
- And of course, inference speed — we had to optimize for time without sacrificing output quality.

## Accomplishments that we're proud of

- We created a fully functional sketch-to-3D pipeline from scratch during the hackathon.
- Our ControlNet integration works reliably with user-provided input, even if the sketch is rough.
- We made the interface intuitive enough that anyone could try it, not just developers or designers.

## What we learned...

- Prompt engineering for generative models is basically a superpower.
- Clean input matters way more than we expected when working with AI pipelines.
- Even with powerful models, postprocessing and constraints make a huge difference in real-world results.
- Collaborating across frontend, backend, and ML workflows was tough at times but made the project way stronger.

## What's next for Lazy3D

We want to expand Lazy3D into a full creative platform. Next steps include:

- Adding more ControlNet modes (like depth or segmentation) to improve shape accuracy
- Giving users more feedback or guidance during sketching
- Letting people edit or animate the generated 3D models directly in the browser
- And maybe eventually supporting rigging or VR export options

But really, this started with a simple goal — make 3D creation lazy — and that’s still what drives us.
