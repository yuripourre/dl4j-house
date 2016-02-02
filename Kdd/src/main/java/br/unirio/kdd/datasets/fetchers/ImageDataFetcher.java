package br.unirio.kdd.datasets.fetchers;

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

import br.unirio.kdd.base.ImageDatasetLoader;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.List;


/**
 * Based on Data fetcher for the LFW faces dataset
 *
 * @author Adam Gibson
 */
public class ImageDataFetcher extends BaseDataFetcher {

    /**
     *
     */
    private static final long serialVersionUID = -7473748140401804666L;
    private ImageDatasetLoader loader;

    public ImageDataFetcher(int imageWidth, int imageHeight, int numChannels) {
        try {
            loader = new ImageDatasetLoader(imageWidth, imageHeight);
            loader.getIfNotExists();

            inputColumns = loader.getNumPixelColumns() * numChannels;
            numOutcomes = loader.getNumTags();
            totalExamples = loader.getNumTags()*loader.NUM_IMAGES_BY_TAG;
        } catch (Exception e) {
            throw new IllegalStateException("Unable to fetch images", e);
        }
    }

    @Override
    public void fetch(int numExamples) {
        if (!hasMore())
            throw new IllegalStateException("Unable to getFromOrigin more; there are no more images");

        //we need to ensure that we don't overshoot the number of examples total
        List<DataSet> toConvert = new ArrayList<>();

        for (int i = 0; i < numExamples; i++, cursor++) {
            if (!hasMore())
                break;
            toConvert.add(loader.getDataFor(cursor));
        }

        initializeCurrFromList(toConvert);
    }

    @Override
    public DataSet next() {
        DataSet next = super.next();
        return next;
    }

}