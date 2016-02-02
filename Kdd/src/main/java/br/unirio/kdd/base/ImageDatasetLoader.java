/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package br.unirio.kdd.base;

import org.deeplearning4j.util.ArchiveUtils;
import org.deeplearning4j.util.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Loads LFW faces data transform. You can customize the size of the images as well
 *
 * @author Adam Gibson
 */
public class ImageDatasetLoader {

    private static final Logger log = LoggerFactory.getLogger(ImageDatasetLoader.class);

    private File baseDir = new File(System.getProperty("user.home"));
    public final static String DATABASE_NAME = "house-bd";
    private File imagesDir = new File(baseDir, DATABASE_NAME);

    private int numTags;
    private int numPixelColumns;
    private ImageLoader loader;
    private List<String> images = new ArrayList<>();
    private List<String> tags = new ArrayList<>();

    public final static int NUM_IMAGES_BY_TAG = 110;

    public ImageDatasetLoader(int imageWidth, int imageHeight) {
        loader = new ImageLoader(imageWidth, imageHeight);
    }

    /*
     * Load files
     */
    public void getIfNotExists() throws Exception {
        tags.add("bed");
        tags.add("toilet_bowl");
        tags.add("chair");
        //tags.add("refrigerator");
        tags.add("couch");

        for (String tag : tags) {
            String path = imagesDir.getAbsolutePath() + File.separator + tag + File.separator;

            System.out.println(path);

            for (int i = 1; i <= NUM_IMAGES_BY_TAG; i++) {
                images.add(path + tag + i + ".jpg");
            }
        }

        File firstImage = imagesDir.listFiles()[0].listFiles()[0];

        //number of input neurons
        numPixelColumns = ArrayUtil.flatten(loader.fromFile(firstImage)).length;

        numTags = tags.size();
    }

    public DataSet convertListPairs(List<DataSet> images) {
        INDArray inputs = Nd4j.create(images.size(), numPixelColumns);
        INDArray outputs = Nd4j.create(images.size(), numTags);

        for (int i = 0; i < images.size(); i++) {
            inputs.putRow(i, images.get(i).getFeatureMatrix());
            outputs.putRow(i, images.get(i).getLabels());
        }
        return new DataSet(inputs, outputs);
    }

    public DataSet getDataFor(int i) {
        File image = new File(images.get(i));
        int outcome = i / NUM_IMAGES_BY_TAG;
        try {
            return new DataSet(loader.asRowVector(image), FeatureUtil.toOutcomeVector(outcome, tags.size()));
        } catch (Exception e) {
            throw new IllegalStateException("Unable to getFromOrigin data for image " + i + " for path " + images.get(i));
        }
    }

    /**
     * Get the first num found images
     *
     * @param num the number of images to getFromOrigin
     * @return
     * @throws Exception
     */
    public List<DataSet> getFeatureMatrix(int num) throws Exception {
        List<DataSet> ret = new ArrayList<>(num);
        File[] files = imagesDir.listFiles();
        int label = 0;
        for (File file : files) {
            ret.addAll(getImages(label, file));
            label++;
            if (ret.size() >= num)
                break;
        }
        return ret;
    }

    public DataSet getAllImagesAsMatrix() throws Exception {
        List<DataSet> images = getImagesAsList();
        return convertListPairs(images);
    }

    public DataSet getAllImagesAsMatrix(int numRows) throws Exception {
        List<DataSet> images = getImagesAsList().subList(0, numRows);
        return convertListPairs(images);
    }

    public List<DataSet> getImagesAsList() throws Exception {
        List<DataSet> list = new ArrayList<>();
        File[] dirs = imagesDir.listFiles();
        for (int i = 0; i < dirs.length; i++) {
            list.addAll(getImages(i, dirs[i]));
        }
        return list;
    }

    public List<DataSet> getImages(int label, File file) throws Exception {
        File[] images = file.listFiles();
        List<DataSet> ret = new ArrayList<>();
        for (File f : images)
            ret.add(fromImageFile(label, f));
        return ret;
    }

    public DataSet fromImageFile(int label, File image) throws Exception {
        INDArray outcome = FeatureUtil.toOutcomeVector(label, numTags);
        INDArray image2 = ArrayUtil.toNDArray(loader.flattenedImageFromFile(image));
        return new DataSet(image2, outcome);
    }

    public void untarFile(File baseDir, File tarFile) throws IOException {

        log.info("Untaring File: " + tarFile.toString());

        ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath(), baseDir.getAbsolutePath());

    }

    public int getNumTags() {
        return numTags;
    }

    public int getNumPixelColumns() {
        return numPixelColumns;
    }

}