package Euclidean_Distance_Transform_MT;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.Prefs;
import ij.gui.GenericDialog;
import ij.measure.Calibration;
import ij.plugin.filter.PlugInFilter;
import static ij.plugin.filter.PlugInFilter.DOES_8G;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

/*
 * Bob Dougherty 8/8/2006
 * Saito-Toriwaki algorithm for Euclidian Distance Transformation.
 * Direct application of Algorithm 1.
 * Version S1A: lower memory usage.
 * Version S1A.1 A fixed indexing bug for 666-bin data set
 * Version S1A.2 Aug. 9, 2006. Changed noResult value.
 * Version S1B Aug. 9, 2006. Faster.
 * Version S1B.1 Sept. 6, 2006. Changed comments.
 * Version S1C Oct. 1, 2006. Option for inverse case.
 * Fixed inverse behavior in y and z directions.
 * Version D July 30, 2007. Multithread processing for step 2.
 *
 * This version assumes the input stack is already in memory, 8-bit, and
 * outputs to a new 32-bit stack. Versions that are more stingy with memory
 * may be forthcoming.
 *
 * License:
 * Copyright (c) 2006, OptiNav, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * Neither the name of OptiNav, Inc. nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

 /*
 * saito toriwaki euclidean distance transform algorithm
 */
public final class Euclidean_Distance_Transform_MT implements PlugInFilter
{
    private ImagePlus image;

    private byte[][] data;

    private int imageWidth;

    private int imageHeight;

    private int imageDepth;

    private int threshold;

    private boolean inverse;

    private float[][] edt;

    private final int numberOfThreads = Runtime.getRuntime().availableProcessors();

    private double squaredVoxelWidth;

    private double squaredVoxelHeight;

    private double squaredVoxelDepth;

    @Override
    public int setup(final String arg,
                     final ImagePlus imp)
    {
        image = imp;
        return DOES_8G;
    }

    @Override
    public void run(ImageProcessor ip)
    {
        final ImageStack stack = image.getStack();
        imageWidth = stack.getWidth();
        imageHeight = stack.getHeight();
        imageDepth = image.getStackSize();

        final Calibration cal = image.getCalibration();
        squaredVoxelWidth = cal.pixelWidth * cal.pixelWidth;
        squaredVoxelHeight = cal.pixelHeight * cal.pixelHeight;
        squaredVoxelDepth = cal.pixelDepth * cal.pixelDepth;

        if (!doDialog())
        {
            return;
        }
        //Create references to input data
        data = new byte[imageDepth][];
        for (int k = 0; k < imageDepth; k++)
        {
            data[k] = (byte[]) stack.getPixels(k + 1);
        }
        //Create 32 bit floating point stack for output, s.  Will also use it for g in Transormation 1.
        ImageStack sStack = new ImageStack(imageWidth, imageHeight);
        edt = new float[imageDepth][];
        for (int k = 0; k < imageDepth; k++)
        {
            ImageProcessor ipk = new FloatProcessor(imageWidth, imageHeight);
            sStack.addSlice(null, ipk);
            edt[k] = (float[]) ipk.getPixels();
        }
        float[] edtSlice;
        //Transformation 1.  Use s to store g.
        IJ.showStatus("EDT transformation 1/3");
        FirstStep[] s1t = new FirstStep[numberOfThreads];
        for (int thread = 0; thread < numberOfThreads; thread++)
        {
            s1t[thread] = new FirstStep(thread);
            s1t[thread].start();
        }
        try
        {
            for (int thread = 0; thread < numberOfThreads; thread++)
            {
                s1t[thread].join();
            }
        }
        catch (InterruptedException ie)
        {
            IJ.error("A thread was interrupted in step 1 .");
        }
        //Transformation 2.  g (in s) -> h (in s)
        IJ.showStatus("EDT transformation 2/3");
        SecondStep[] s2t = new SecondStep[numberOfThreads];
        for (int thread = 0; thread < numberOfThreads; thread++)
        {
            s2t[thread] = new SecondStep(thread);
            s2t[thread].start();
        }
        try
        {
            for (int thread = 0; thread < numberOfThreads; thread++)
            {
                s2t[thread].join();
            }
        }
        catch (InterruptedException ie)
        {
            IJ.error("A thread was interrupted in step 2 .");
        }
        //Transformation 3. h (in s) -> s
        IJ.showStatus("EDT transformation 3/3");
        ThirdStep[] s3t = new ThirdStep[numberOfThreads];
        for (int thread = 0; thread < numberOfThreads; thread++)
        {
            s3t[thread] = new ThirdStep(thread);
            s3t[thread].start();
        }
        try
        {
            for (int thread = 0; thread < numberOfThreads; thread++)
            {
                s3t[thread].join();
            }
        }
        catch (InterruptedException ie)
        {
            IJ.error("A thread was interrupted in step 3 .");
        }
        //Find the largest distance for scaling
        //Also fill in the background values.
        float distMax = 0;
        int wh = imageWidth * imageHeight;
        for (int k = 0; k < imageDepth; k++)
        {
            edtSlice = edt[k];
            for (int ind = 0; ind < wh; ind++)
            {
                if (((data[k][ind] & 255) < threshold) ^ inverse)
                {
                    edtSlice[ind] = 0;
                }
                else
                {
                    final float dist = (float) Math.sqrt(edtSlice[ind]);
                    edtSlice[ind] = dist;
                    if (dist > distMax)
                    {
                        distMax = dist;
                    }
                }
            }
        }

        IJ.showProgress(1.0);
        IJ.showStatus("Done");
        String title = stripExtension(image.getTitle());
        ImagePlus impOut = new ImagePlus(title + "EDT", sStack);
        impOut.getProcessor().setMinAndMax(0, distMax);
        impOut.show();
        IJ.run("Fire");
    }

    //Modified from ImageJ code by Wayne Rasband
    private String stripExtension(String name)
    {
        if (name != null)
        {
            int dotIndex = name.lastIndexOf(".");
            if (dotIndex >= 0)
            {
                name = name.substring(0, dotIndex);
            }
        }
        return name;
    }

    private boolean doDialog()
    {
        threshold = (int) Prefs.get("edtS1.thresh", 128);
        inverse = Prefs.get("edtS1.inverse", false);

        final GenericDialog gd = new GenericDialog("EDT...", IJ.getInstance());
        gd.addNumericField("Threshold (1 to 255; value < thresh is background)", threshold, 0);
        gd.addCheckbox("Inverse case (background when value >= thresh)", inverse);
        gd.showDialog();
        if (gd.wasCanceled())
        {
            return false;
        }
        threshold = (int) gd.getNextNumber();
        inverse = gd.getNextBoolean();

        Prefs.set("edtS1.thresh", threshold);
        Prefs.set("edtS1.inverse", inverse);

        return true;
    }

    private final class FirstStep extends Thread
    {
        private final int zStart;

        FirstStep(final int start)
        {
            zStart = start;
        }

        @Override
        public void run()
        {
            for (int z = zStart; z < imageDepth; z += numberOfThreads)
            {
                IJ.showProgress(z / (1. * imageDepth));
                final float[] edtSlice = edt[z];
                final byte[] dataSlice = data[z];
                for (int y = 0; y < imageHeight; y++)
                {
                    final boolean[] background = new boolean[imageWidth];
                    for (int x = 0; x < imageWidth; x++)
                    {
                        background[x] = ((dataSlice[x + imageWidth * y] & 255) < threshold) ^ inverse;
                    }
                    for (int x1 = 0; x1 < imageWidth; x1++)
                    {
                        int min = Integer.MAX_VALUE;
                        for (int x2 = x1; x2 < imageWidth; x2++)
                        {
                            if (background[x2])
                            {
                                int test = x1 - x2;
                                test *= test;
                                min = test;
                                break;
                            }
                        }
                        for (int x2 = x1 - 1; x2 >= 0; x2--)
                        {
                            if (background[x2])
                            {
                                int test = x1 - x2;
                                test *= test;
                                if (test < min)
                                {
                                    min = test;
                                }
                                break;
                            }
                        }
                        edtSlice[x1 + imageWidth * y] = (float) (min * squaredVoxelWidth);
                    }
                }
            }
        }
    }

    private final class SecondStep extends Thread
    {
        private final int zStart;

        SecondStep(final int start)
        {
            zStart = start;
        }

        @Override
        public void run()
        {
            for (int z = zStart; z < imageDepth; z += numberOfThreads)
            {
                IJ.showProgress(z / (1. * imageDepth));
                final float[] edtSlice = edt[z];
                for (int x = 0; x < imageWidth; x++)
                {
                    boolean nonempty = false;
                    final float[] tmp1 = new float[imageHeight];
                    for (int y = 0; y < imageHeight; y++)
                    {
                        tmp1[y] = edtSlice[x + imageWidth * y];
                        if (tmp1[y] > 0)
                        {
                            nonempty = true;
                        }
                    }
                    if (nonempty)
                    {
                        final float[] tmp2 = new float[imageHeight];
                        for (int y1 = 0; y1 < imageHeight; y1++)
                        {
                            float min = Float.MAX_VALUE;
                            int delta = y1;
                            for (int y2 = 0; y2 < imageHeight; y2++)
                            {
                                float test = (float) (tmp1[y2] + delta * delta-- * squaredVoxelHeight);
                                if (test < min)
                                {
                                    min = test;
                                }
                            }
                            tmp2[y1] = min;
                        }
                        for (int y = 0; y < imageHeight; y++)
                        {
                            edtSlice[x + imageWidth * y] = tmp2[y];
                        }
                    }
                }
            }
        }
    }

    private final class ThirdStep extends Thread
    {
        private final int yStart;

        ThirdStep(final int start)
        {
            yStart = start;
        }

        @Override
        public void run()
        {
            for (int y = yStart; y < imageHeight; y += numberOfThreads)
            {
                IJ.showProgress(y / (1. * imageHeight));
                for (int x = 0; x < imageWidth; x++)
                {
                    boolean nonempty = false;
                    float[] tmp1 = new float[imageDepth];
                    for (int z = 0; z < imageDepth; z++)
                    {
                        tmp1[z] = edt[z][x + imageWidth * y];
                        if (((int) tmp1[z]) > 0)
                        {
                            nonempty = true;
                        }
                    }
                    if (nonempty)
                    {
                        int zStart = 0;
                        while ((zStart < (imageDepth - 1)) && ((int) tmp1[zStart] == 0))
                        {
                            zStart++;
                        }
                        if (zStart > 0)
                        {
                            zStart--;
                        }
                        int zStop = imageDepth - 1;
                        while ((zStop > 0) && ((int) tmp1[zStop] == 0))
                        {
                            zStop--;
                        }
                        if (zStop < (imageDepth - 1))
                        {
                            zStop++;
                        }
                        float[] tmp2 = new float[imageDepth];
                        for (int z1 = 0; z1 < imageDepth; z1++)
                        {
                            //Limit to the non-background to save time,
                            if (((data[z1][x + imageWidth * y] & 255) >= threshold) ^ inverse)
                            {
                                float min = Float.MAX_VALUE;
                                int zBegin = zStart;
                                int zEnd = zStop;
                                if (zBegin > z1)
                                {
                                    zBegin = z1;
                                }
                                if (zEnd < z1)
                                {
                                    zEnd = z1;
                                }
                                int delta = z1 - zBegin;
                                for (int z2 = zBegin; z2 <= zEnd; z2++)
                                {
                                    final float test = (float) (tmp1[z2] + delta * delta-- * squaredVoxelDepth);
                                    if (test < min)
                                    {
                                        min = test;
                                    }
                                }
                                tmp2[z1] = min;
                            }
                        }
                        for (int z = 0; z < imageDepth; z++)
                        {
                            edt[z][x + imageWidth * y] = tmp2[z];
                        }
                    }
                }
            }
        }
    }
}
