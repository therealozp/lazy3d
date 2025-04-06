import React, { Suspense, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import {
	Html,
	OrbitControls,
	useGLTF,
	Environment,
	ContactShadows,
	PerspectiveCamera,
} from '@react-three/drei';

function Model({ url }) {
	const gltf = useGLTF(url);
	return <primitive object={gltf.scene} dispose={null} />;
}

export default function ModelViewer({ modelUrl }) {
	const memoizedUrl = useMemo(() => modelUrl, [modelUrl]);

	return (
		<div className="w-screen h-screen">
				<Canvas
					gl={{
						antialias: true,
						powerPreference: 'high-performance',
						alpha: false,
					}}
					dpr={[1, 2]} // Adjust based on device capabilities
					performance={{ min: 0.5 }} // Performance throttling
					shadows
				>
					<color attach="background" args={['#fafafa']} />
					<fog attach="fog" args={['#fafafa', 5, 20]} />
					<PerspectiveCamera makeDefault position={[0, 1, 5]} fov={45} />

					<gridHelper
						args={[100, 100, '#2e2f30', '#828399']}
						position={[0, -0.5, 0]}
					/>
					{/* <axesHelper args={[2]} /> */}

					<ambientLight intensity={0.5} />
					<directionalLight
						position={[5, 5, 5]}
						intensity={1}
						castShadow
						shadow-mapSize={[1024, 1024]}
					/>
					<Suspense fallback={<Html><div style={{ color: 'black' }}>LOADING...</div></Html>}>
						<Model url={memoizedUrl} />
						<Environment preset="city" />
					</Suspense>

					<ContactShadows
						position={[0, -0.5, 0]}
						opacity={0.1}
						scale={10}
						blur={1.5}
					/>

					<OrbitControls
						makeDefault
						minPolarAngle={0}
						maxPolarAngle={Math.PI / 2}
						enableDamping
						dampingFactor={0.05}
					/>
				</Canvas>
		</div>
	);
}
