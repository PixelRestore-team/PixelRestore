
const $ = id => document.getElementById(id);

let generatedMask = null;
let previewURL = null;
let damagedPreviewURL = null;
let maskIsValid = false;

// Create a preview area for the uploaded damaged image
const damagedPreview = document.createElement('div');

damagedPreview.className = 'image-preview';
damagedPreview.style.marginTop = '15px';
damagedPreview.style.textAlign = 'center';

$('masked').parentElement.appendChild(damagedPreview);


// Check backend and model availability
async function health() {
    try {
        const r = await fetch('/api/health');
        const x = await r.json();

        $('health').textContent =
            x.srn_model && x.trn_model
                ? 'models ready'
                : 'models missing';

    } catch (e) {
        $('health').textContent = 'backend offline';
    }
}


// Convert a base64 image into a result tile
function tile(title, b64) {
    return `
        <div class="tile">
            <img src="data:image/png;base64,${b64}" alt="${title}">
            <div>${title}</div>
        </div>
    `;
}


// Display uploaded damaged image
function showDamagedPreview(file) {

    if (damagedPreviewURL) {
        URL.revokeObjectURL(damagedPreviewURL);
    }

    damagedPreviewURL = URL.createObjectURL(file);

    damagedPreview.innerHTML = '';

    const heading = document.createElement('p');
    heading.textContent = 'Uploaded Image Preview';
    heading.style.fontWeight = '600';

    const img = document.createElement('img');
    img.src = damagedPreviewURL;
    img.alt = 'Uploaded damaged image';

    img.style.width = '100%';
    img.style.maxHeight = '220px';
    img.style.objectFit = 'contain';
    img.style.borderRadius = '10px';

    damagedPreview.appendChild(heading);
    damagedPreview.appendChild(img);
}


// Generate a binary mask from white regions
async function generateMask(file) {

    const image = await createImageBitmap(file);

    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;

    const ctx = canvas.getContext('2d');

    // Fill transparent areas with white
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.drawImage(image, 0, 0);

    const imageData = ctx.getImageData(
        0, 0, canvas.width, canvas.height
    );

    const pixels = imageData.data;

    let missingPixels = 0;
    const totalPixels = canvas.width * canvas.height;

    // White = missing (255), black = known (0)
    for (let i = 0; i < pixels.length; i += 4) {

        const r = pixels[i];
        const g = pixels[i + 1];
        const b = pixels[i + 2];

        const isWhite =
            r >= 245 &&
            g >= 245 &&
            b >= 245;

        const value = isWhite ? 255 : 0;

        pixels[i] = value;
        pixels[i + 1] = value;
        pixels[i + 2] = value;
        pixels[i + 3] = 255;

        if (isWhite) {
            missingPixels++;
        }
    }

    const missingRatio = missingPixels / totalPixels;

    ctx.putImageData(imageData, 0, 0);

    image.close();

    // Reject images with no detected damage or entirely white images
    if (missingRatio <= 0 || missingRatio >= 1) {
        throw new Error(
            'No valid white missing regions were detected. ' +
            'Please upload an image containing both white damaged ' +
            'regions and visible image content.'
        );
    }

    // Convert generated mask into a PNG file
    const blob = await new Promise(resolve => {
        canvas.toBlob(resolve, 'image/png');
    });

    if (!blob) {
        throw new Error('Could not generate the damage mask.');
    }

    generatedMask = new File(
        [blob],
        'generated-mask.png',
        { type: 'image/png' }
    );

    maskIsValid = true;

    // Display generated mask
    if (previewURL) {
        URL.revokeObjectURL(previewURL);
    }

    previewURL = URL.createObjectURL(blob);

    const preview = $('mask-preview');

    preview.innerHTML = '';

    const img = document.createElement('img');
    img.src = previewURL;
    img.alt = 'Automatically generated damage mask';

    img.style.width = '100%';
    img.style.maxHeight = '220px';
    img.style.objectFit = 'contain';
    img.style.borderRadius = '10px';

    preview.appendChild(img);

    return missingRatio;
}


// Handle damaged image upload
$('masked').addEventListener('change', async () => {

    const file = $('masked').files[0];

    generatedMask = null;
    maskIsValid = false;
    $('run').disabled = true;

    if (!file) {
        damagedPreview.innerHTML = '';
        $('mask-preview').textContent =
            'Your generated mask will appear here.';
        return;
    }

    if (!file.type.startsWith('image/')) {
        $('status').classList.remove('hidden');
        $('status').textContent = 'Please select a valid image file.';
        return;
    }

    showDamagedPreview(file);

    $('status').classList.remove('hidden');
    $('status').textContent =
        'Detecting white missing regions and generating mask...';

    try {

        const ratio = await generateMask(file);

        $('status').textContent =
            `Mask generated successfully. ` +
            `${(ratio * 100).toFixed(2)}% of the image is marked as missing.`;

        $('run').disabled = false;

    } catch (e) {

        generatedMask = null;
        maskIsValid = false;

        $('status').textContent = 'Mask generation error: ' + e.message;

        $('mask-preview').textContent =
            'No valid mask generated.';

        if (previewURL) {
            URL.revokeObjectURL(previewURL);
            previewURL = null;
        }
    }
});


// Run reconstruction
$('run').onclick = async () => {

    const masked = $('masked').files[0];

    if (!masked || !generatedMask || !maskIsValid) {
        alert('Please upload a valid damaged image first.');
        return;
    }

    const fd = new FormData();

    fd.append('masked_image', masked);
    fd.append('mask_image', generatedMask);
    fd.append('domain', $('domain').value);

    if ($('gt').files[0]) {
        fd.append('ground_truth', $('gt').files[0]);
    }

    $('run').disabled = true;

    $('status').classList.remove('hidden');
    $('status').textContent =
        'Running SRN + TRN reconstruction and generating semantic guidance...';

    try {

        const r = await fetch('/api/reconstruct', {
            method: 'POST',
            body: fd
        });

        const x = await r.json();

        if (!r.ok) {
            throw new Error(x.detail || 'Reconstruction failed');
        }

        $('status').textContent = 'Reconstruction complete.';

        // Display guidance
        $('guidance').classList.remove('hidden');

        $('source').textContent =
            x.guidance.source === 'openai'
                ? `LLM: ${x.guidance.model}`
                : 'Deterministic fallback';

        $('strategy').textContent = x.guidance.strategy;
        $('rationale').textContent = x.guidance.rationale;

        $('instructions').innerHTML =
            x.guidance.instructions
                .map(i => `<li>${i}</li>`)
                .join('');

        // Display reconstruction results
        $('results').classList.remove('hidden');

        const im = x.images;

        let html =
            tile('Damaged Input', im.masked_input) +
            tile('Generated Mask', im.mask) +
            tile('SRN Output', im.srn_output) +
            tile('TRN Candidate', im.trn_candidate) +
            tile('Final Reconstructed Output', im.final_output);

        if (im.ground_truth) {
            html += tile('Ground Truth', im.ground_truth);
        }

        $('images').innerHTML = html;

        // Display metrics
        const rows = [];

        for (const [name, m] of Object.entries(x.metrics)) {

            if (m) {
                rows.push(`
                    <tr>
                        <td>${name}</td>
                        <td>${m.full_psnr.toFixed(4)}</td>
                        <td>${m.ssim.toFixed(4)}</td>
                        <td>${m.missing_psnr.toFixed(4)}</td>
                    </tr>
                `);
            }
        }

        $('metrics').innerHTML = rows.length
            ? rows.join('')
            : '<tr><td colspan="4">Upload ground truth to compute metrics.</td></tr>';

        $('known').textContent =
            `Known-pixel max delta: ` +
            `${x.metrics.known_pixel_max_delta.toExponential(2)} • ` +
            `${x.model.known_pixels_preserved
                ? 'known pixels preserved exactly'
                : 'known-pixel check failed'}`;

    } catch (e) {

        $('status').textContent = 'Error: ' + e.message;

    } finally {

        $('run').disabled = !maskIsValid;
    }
};


// Initial setup
$('run').disabled = true;
health();
