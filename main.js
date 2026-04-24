const MODEL_DIR = "final_models/EfficientNet_b2_iou_0.251/"
const threshold = 0.50; // model classification threshold
let FILE = "";
const CARD_TEMPLATE = new Image();
CARD_TEMPLATE.src = "card_template.png";
const SET_LOGO = new Image();
SET_LOGO.src = "set_logo.png";

// Source - https://stackoverflow.com/a/45931408
// Posted by clabe45, modified by community. See post 'Timeline' for change history
// Retrieved 2026-04-12, License - CC BY-SA 4.0
window.addEventListener('load', function () {
    document.querySelector('input[type="file"]').addEventListener('change', function () {
        if (this.files && this.files[0]) {
            var img = document.createElement(`img`);
            // onload is changed from stackoverflow source to immediately preprocess the image and run the model
            img.onload = async function () {
                URL.revokeObjectURL(img.src);

                // get 3 crops at extremeties/center and turn into input tensors
                const crops = get_crops(img).map(crop => preprocessImage(img, crop));
                const inputTensors = crops.map(c => {
                    c.tensor = to_tensor(c.canvas, c.ctx);
                    return c;
                });

                const result = await get_result(inputTensors);
                const types = result.type.map(t => t.type).join(" ");
                const filename = document.getElementById("fileinput").value;
                draw_result(result.img.canvas, types, filename);

                for (const crop of crops) {
                    crop.canvas.remove();
                }

                img.remove();
            };

            img.src = URL.createObjectURL(this.files[0]); // set src to blob url
        }
    });
});

const get_result = async function (inputs) {
    // get results for all crops
    let best_scores = [];
    let best_predicted = [];
    let best_avg = 0;
    let best_img = inputs[0];
    for (const image of inputs) {
        const input = image.tensor;
        const result = await session.run({ input: input });
        const logits = result.output.data;

        const scores = Array.from(logits).map((logit, i) => ({
            type: all_types[i],
            score: logit
        }));

        // sort by score descending
        scores.sort((a, b) => b.score - a.score);

        // predicted types above threshold
        const predicted = scores.filter(s => s.score > threshold);
        let avg = 0;
        for (const ctype of scores.slice(0, predicted.length)) {
            avg += ctype.score;
        }

        avg /= predicted.length + 1e-6;

        if (avg >= best_avg) {
            best_scores = scores;
            best_predicted = predicted;
            best_avg = avg;
            best_img = image;
        }
    }

    // display results
    document.getElementById("predicted_types").innerHTML = "types: " + (best_predicted.map(s => s.type).join(" ") || "None");
    document.getElementById("top_5_types").innerHTML = best_scores.slice(0, 5).map(s => `<li>${s.type} (${s.score.toFixed(3)})</li>`).join("");

    return { img: best_img, type: best_predicted };
}

const draw_result = function (img, ctype, name) {
    const c = document.getElementById("result_canvas");
    const ctx = c.getContext("2d");
    
    // Card template + image
    ctx.drawImage(img, c.width * 0.1, c.width * 0.1, c.width * 0.8, c.width * 0.8);
    ctx.drawImage(CARD_TEMPLATE, 0, 0, c.width, c.height);

    // Card name
    ctx.fillStyle = "black";
    ctx.font = "24px Beleren";
    name = name.replace("C:\\fakepath\\", "");
    name = name.split(".");
    name.pop();
    name = name.join(".");
    ctx.fillText(name, 55, 71);

    // Card type-line
    ctx.font = "22px Beleren";
    const typeline = ctype ? "Creature — " + ctype : "Creature";
    ctx.fillText(typeline, 55, 437);

    // set logo
    ctx.drawImage(SET_LOGO, 440, 411, 35, 35);

    // Footer info
    ctx.fillStyle = "white";
    ctx.font = "13px Verdana";
    ctx.fillText("1/∞ R", 39, 692);
    ctx.fillText("IRR · EN 🖋️Mysterious Artist", 39, 710);
    ctx.font = "10px Verdana";
    ctx.fillText("This makes it look more authentic©", 300, 710);
}

let session = null;
let all_types = null;

const init_model = async function () {
    session = await ort.InferenceSession.create(MODEL_DIR + "model.onnx");
    const response = await fetch(MODEL_DIR + "classes.json");
    all_types = await response.json();
}

init_model();

const get_crops = function (img) {
    const w = img.naturalWidth;
    const h = img.naturalHeight;

    if (w > h) {
        // landscape — crop left, center, right
        return ['left', 'center', 'right'];
    } else if (h > w) {
        // portrait — crop top, center, bottom
        return ['top', 'center', 'bottom'];
    } else {
        // already square — just one crop
        return ['center'];
    }
}

const preprocessImage = function (img, crop_position = "center") {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');

    const scale = 256 / Math.min(img.naturalWidth, img.naturalHeight);
    const scaledW = img.naturalWidth * scale;
    const scaledH = img.naturalHeight * scale;

    let offsetX, offsetY;

    if (crop_position === 'center') {
        offsetX = (scaledW - 224) / 2;
        offsetY = (scaledH - 224) / 2;
    } else if (crop_position === 'left' || crop_position === 'top') {
        offsetX = 0;
        offsetY = 0;
    } else if (crop_position === 'right' || crop_position === 'bottom') {
        offsetX = scaledW - 224;
        offsetY = scaledH - 224;
    }

    ctx.drawImage(img, -offsetX, -offsetY, scaledW, scaledH);

    return { canvas, ctx };
}

const to_tensor = function (c, ctx) {
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const pixels = imageData.data; // RGBA, values 0-255

    // ImageNet normalisation constants
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // rearrange from RGBA interleaved to CHW (channel, height, width)
    const tensor = new Float32Array(3 * 224 * 224);
    for (let i = 0; i < 224 * 224; i++) {
        for (let c = 0; c < 3; c++) {
            tensor[c * 224 * 224 + i] = (pixels[i * 4 + c] / 255 - mean[c]) / std[c];
        }
    }

    return new ort.Tensor('float32', tensor, [1, 3, 224, 224]);
}