# Offline LLM · Llama 3.2 1B · WebGPU

A tiny, dependency-free web app that runs **Llama 3.2 1B (4-bit)** entirely in
your browser on the **GPU via WebGPU** — no server, no API key, and **no network
once the model is cached**. Open the page, click *Load model*, and chat.

It uses [MLC **WebLLM**](https://github.com/mlc-ai/web-llm), which compiles LLMs
to WebGPU shaders and runs inference client-side. Weights are downloaded once
(~0.8 GB for the 1B model) and cached in the browser, so you can turn off Wi-Fi
and it keeps working.

## Why this is interesting

- **Fully on-device.** Your prompts never leave the machine.
- **Offline.** After the first load, airplane mode is fine.
- **Real hardware acceleration.** WebGPU maps to Metal on Apple devices, so it
  uses the GPU — the most powerful ML unit a browser can reach (the Neural
  Engine is *not* exposed to web code).

## Run it

WebGPU requires a secure context, so serve over `localhost` (not `file://`).
Any static server works — pick one:

```bash
# Python (built in on macOS/Linux)
cd offline-llm-webgpu
python3 -m http.server 8000
# → open http://localhost:8000

# …or Node
npx serve .

# …or VS Code "Live Server" extension
```

Then open the URL, click **Load model**, wait for the one-time download, and chat.

### Trying it on an iPhone 15

1. Serve the folder on your computer (commands above).
2. Make sure the phone is on the **same Wi-Fi**, and open
   `http://<your-computer-ip>:8000` in Safari.
   - For a *real* offline test you'll want HTTPS (e.g. tunnel via
     `cloudflared` / `ngrok`, or host it on any static host like GitHub Pages),
     because Safari only enables WebGPU in a secure context.
3. iOS needs **17.4+**. If WebGPU is off, enable it in
   **Settings → Safari → Advanced → Feature Flags → WebGPU**.
4. The base iPhone 15 (A16, 6 GB RAM) comfortably runs the **1B** model. The
   **3B** option is included but pushes Safari's per-tab memory limits — treat
   it as experimental on a phone.

## Hosting (so it's truly portable / offline on mobile)

This is a fully static site. Drop the three files on any static host:

- **GitHub Pages**, Netlify, Cloudflare Pages, Vercel — all work with zero config.
- HTTPS is provided automatically, which satisfies WebGPU's secure-context rule.

## Design — Apple Liquid Glass

The UI is a faithful recreation of Apple's **Liquid Glass** material (iOS 26 /
WWDC 2025). Every control is real glass:

- **Translucent body** — `backdrop-filter` blur + saturation, never opaque.
- **Edge lensing** — an SVG displacement filter (`#lg-lens`) refracts the
  backdrop at the rim, with a bright glass bevel.
- **Motion-tracked specular highlights** — the glint follows a virtual light
  source: your **pointer** on desktop, the **gyroscope** on mobile (tap once to
  grant motion access on iOS).
- **Liquid spring animations** — press squash, bubble pop-in, panel morphs, an
  animated wallpaper the glass refracts, and an animated rim wobble.
- **Type** follows Apple's 17px body baseline for comfortable mobile reading.
- Honors **`prefers-reduced-motion`** for accessibility.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Markup, glass material layers, SVG lensing filters |
| `style.css`  | Liquid Glass material + animations (mobile-first, 17px type) |
| `app.js`     | WebLLM engine, streaming, UI logic |
| `glass.js`   | Liquid Glass interaction layer (light tracking, gyro, press, morphs) |

## Model options (in the *Options* panel)

| Build | Notes |
|-------|-------|
| `Llama-3.2-1B-Instruct-q4f32_1-MLC` | **Default.** Best compatibility, ~0.8 GB. |
| `Llama-3.2-1B-Instruct-q4f16_1-MLC` | Smaller/faster, needs f16 GPU support. |
| `Llama-3.2-3B-Instruct-q4f32_1-MLC` | Smarter, heavier (~2 GB) — desktop-friendly, tight on mobile. |

The full WebLLM model list (Qwen, Phi, Gemma, etc.) is available if you swap the
`value` in the `<select>` — see the
[prebuilt model list](https://github.com/mlc-ai/web-llm/blob/main/src/config.ts).

## Notes & limitations

- First load needs internet (to fetch weights); everything after is local.
- The browser can use **GPU + CPU only** — the iPhone's Neural Engine is not
  reachable from web code. For that you'd need a native Core ML / MLX app.
- Performance scales with GPU + memory. Expect smooth interactive speeds for 1B.
