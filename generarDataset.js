const fs = require("fs");

const VAL_SIZE = 0.2;
const IMG_TRAIN_DIR = "./dataset/images/train";
const LABEL_TRAIN_DIR = "./dataset/labels/train";
const IMG_VAL_DIR = "./dataset/images/val";
const LABEL_VAL_DIR = "./dataset/labels/val";

const eliminarLabelsRaras = () => {
    fs.readdirSync(LABEL_TRAIN_DIR).forEach(label => {
        if(label.match(/(_aug[0-9]+\.txt)$/)) {
            fs.rmSync(LABEL_TRAIN_DIR + "/" + label);
        }
    });
}

//Chequear si existe label para imagen
const checkSiLabelExiste = () => {
    const imgs = fs.readdirSync(IMG_TRAIN_DIR);
    imgs.forEach(img => {
        const [imgName] = img.split(".");
        const labelPath = LABEL_TRAIN_DIR + "/" + imgName + ".txt";
        if(!fs.existsSync(labelPath)) {
            console.log("Eliminando...", imgName);
            fs.rmSync(IMG_TRAIN_DIR + "/" + img);
        }
    });
}

const separarTrainVal = () => {
    const imgs = fs.readdirSync(IMG_TRAIN_DIR);
    const IDX = Math.floor(imgs.length * VAL_SIZE);
    const imgFiltered = imgs.filter((_, i) => i <= IDX);
    imgFiltered.forEach((img, i) => {
        const [imgName] = img.split(".");
        const labelPath = LABEL_TRAIN_DIR + "/" + imgName + ".txt";
        console.log("Moviendo", imgName);
        const imgPath = IMG_TRAIN_DIR + "/" + img;
        fs.copyFileSync(imgPath, IMG_VAL_DIR + "/" + img);
        fs.copyFileSync(labelPath, LABEL_VAL_DIR + "/" + imgName + ".txt");
        fs.rmSync(imgPath);
        fs.rmSync(labelPath)
    });
}

eliminarLabelsRaras();
checkSiLabelExiste();
separarTrainVal();