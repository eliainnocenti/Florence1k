# Images

This directory contains images used in the Florence 1k dataset for monument recognition.

## Image Sources

The images used in this dataset are primarily sourced from user uploads on Google Maps 
and Google Images. These images represent various monuments and landmarks in Florence, 
Italy, captured by tourists and locals alike.

## Image Details

### Format

- All images are in JPEG format
- Images are in RGB color space
- Resolution varies, but all images have a minimum dimension of `50000` pixels (width x height)

### Content

The images depict 12 famous monuments in Florence:

1. Cattedrale di Santa Maria del Fiore (Duomo di Firenze)
2. Battistero di San Giovanni
3. Campanile di Giotto
4. Galleria degli Uffizi
5. Loggia dei Lanzi
6. Palazzo Vecchio
7. Ponte Vecchio
8. Basilica di Santa Croce
9. Palazzo Pitti
10. Piazzale Michelangelo
11. Basilica di Santa Maria Novella
12. Basilica di San Miniato al Monte

### Variety

The dataset includes a diverse range of images for each monument:

- Different angles and perspectives
- Various lighting conditions (day, night, sunny, cloudy)
- Different seasons
- With and without crowds
- Close-ups and distant views

## Usage Rights

While these images are sourced from public uploads, it's important to note:

1. The images are used here for research and educational purposes under fair use.
2. If you plan to use this dataset for commercial purposes, you should seek appropriate permissions.
3. We do not claim ownership of these images. All rights belong to their respective owners.

## File Naming Convention

Images are named using the following convention:

```
florence_<monument_name>_<image_id>.jpg
```

For example:

- `florence_santamariadelfiore_0097.jpg`
- `florence_pontevecchio_0052.jpg`

## Annotations

Each image in this directory has a corresponding annotation file in the `annotations` directory. 
These files contain bounding box coordinates and class labels for the monuments in each image.

## Ethical Considerations

We have made efforts to ensure that the dataset:

- Does not contain any inappropriate or offensive content
- Does not include identifiable individuals
- Does not violate any privacy or copyright laws
- Represents a diverse range of perspectives and conditions

If you notice any images that raise ethical concerns, please contact the dataset maintainers.

## Contributing

If you have high-quality images of Florence monuments that you own and would like to contribute 
to this dataset, please see our contribution guidelines in the CONTRIBUTING.md file.

## Contact

For any questions or concerns regarding the images in this dataset, please contact the dataset 
maintainers at [elia.innocenti@edu.unifi.it](mailto:elia.innocenti@edu.unifi.it).