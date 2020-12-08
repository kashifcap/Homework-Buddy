# -*- coding: utf-8 -*-
"""
Created on Thu Nov  26 15:13:36 2020

@author: Pratik
"""


import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model_path', type=str, default=os.path.join('pretrained', 'model-29'))
parser.add_argument('--style', dest='style', type=int, default=None)
parser.add_argument('--bias', dest='bias', type=float, default=1.)
parser.add_argument('--force', dest='force', action='store_true', default=False)
parser.add_argument('--noinfo', dest='info', action='store_false', default=True)
parser.add_argument('--save', dest='save', type=str, default=None)
args = parser.parse_args()


def sample(e, mu1, mu2, std1, std2, rho):
    cov = np.array([[std1 * std1, std1 * std2 * rho],
                    [std1 * std2 * rho, std2 * std2]])
    mean = np.array([mu1, mu2])

    x, y = np.random.multivariate_normal(mean, cov)
    end = np.random.binomial(1, e)
    return np.array([x, y, end])


def split_strokes(points):
    points = np.array(points)
    strokes = []
    b = 0
    for e in range(len(points)):
        if points[e, 2] == 1.:
            strokes += [points[b: e + 1, :2].copy()]
            b = e + 1
    return strokes


def cumsum(points):
    sums = np.cumsum(points[:, :2], axis=0)
    return np.concatenate([sums, points[:, 2:]], axis=1)


def sample_text(sess, args_text, translation, style=None):
    fields = ['coordinates', 'sequence', 'bias', 'e', 'pi', 'mu1', 'mu2', 'std1', 'std2',
              'rho', 'window', 'kappa', 'phi', 'finish', 'zero_states']
    vs = namedtuple('Params', fields)(
        *[tf.compat.v1.get_collection(name)[0] for name in fields]
    )

    text = np.array([translation.get(c, 0) for c in args_text])
    coord = np.array([0., 0., 1.])
    coords = [coord]

    # Prime the model with the author style if requested
    prime_len, style_len = 0, 0
    if style is not None:
        # Priming consist of joining to a real pen-position and character sequences the synthetic sequence to generate
        #   and set the synthetic pen-position to a null vector (the positions are sampled from the MDN)
        style_coords, style_text = style
        prime_len = len(style_coords)
        style_len = len(style_text)
        prime_coords = list(style_coords)
        coord = prime_coords[0] # Set the first pen stroke as the first element to process
        text = np.r_[style_text, text] # concatenate on 1 axis the prime text + synthesis character sequence
        sequence_prime = np.eye(len(translation), dtype=np.float32)[style_text]
        sequence_prime = np.expand_dims(np.concatenate([sequence_prime, np.zeros((1, len(translation)))]), axis=0)
        
    sequence = np.eye(len(translation), dtype=np.float32)[text]
    sequence = np.expand_dims(np.concatenate([sequence, np.zeros((1, len(translation)))]), axis=0)

    phi_data, window_data, kappa_data, stroke_data = [], [], [], []
    sess.run(vs.zero_states)
    sequence_len = len(args_text) + style_len
    for s in range(1, 60 * sequence_len + 1):
        is_priming = s < prime_len

        print('\r[{:5d}] sampling... {}'.format(s, 'priming' if is_priming else 'synthesis'), end='')

        e, pi, mu1, mu2, std1, std2, rho, \
        finish, phi, window, kappa = sess.run([vs.e, vs.pi, vs.mu1, vs.mu2,
                                               vs.std1, vs.std2, vs.rho, vs.finish,
                                               vs.phi, vs.window, vs.kappa],
                                              feed_dict={
                                                  vs.coordinates: coord[None, None, ...],
                                                  vs.sequence: sequence_prime if is_priming else sequence,
                                                  vs.bias: args.bias
                                              })

        if is_priming:
            # Use the real coordinate if priming
            coord = prime_coords[s]
        else:
            # Synthesis mode
            phi_data += [phi[0, :]]
            window_data += [window[0, :]]
            kappa_data += [kappa[0, :]]
            # ---
            g = np.random.choice(np.arange(pi.shape[1]), p=pi[0])
            coord = sample(e[0, 0], mu1[0, g], mu2[0, g],
                           std1[0, g], std2[0, g], rho[0, g])
            coords += [coord]
            stroke_data += [[mu1[0, g], mu2[0, g], std1[0, g], std2[0, g], rho[0, g], coord[2]]]

            if not args.force and finish[0, 0] > 0.8:
                print('\nFinished sampling!\n')
                break

    coords = np.array(coords)
    coords[-1, 2] = 1.

    return phi_data, window_data, kappa_data, stroke_data, coords


def main():
    with open(os.path.join('data', 'translation.pkl'), 'rb') as file:
        translation = pickle.load(file)
    #rev_translation for b->a mapped dictionary
    rev_translation = {v: k for k, v in translation.items()}
    charset = [rev_translation[i] for i in range(len(rev_translation))]
    charset[0] = ''
    coun = 0
    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 0}
    )
    '''Reads text from text.txt file and takes 8 words at a time 
       generates line by line
       Future version will have an arguments for no. of words per line
       and no. of line per page of pdf.'''
       
    goku = open("text.txt", 'r')
    args_text = goku.read()
    tempo = args_text.split()
    for i in range(0, len(tempo), 8):
        st = str("")
        if i+8 > len(tempo):
            last = len(tempo)
        else:
            last = i+8
        for j in range(i,last, 1):
            st = str(st) + str(tempo[j]) + " "
        with tf.compat.v1.Session(config=config) as sess:
            saver = tf.compat.v1.train.import_meta_graph(args.model_path + '.meta')
            saver.restore(sess, args.model_path)
            style = None
            if args.style is not None:
                style = None
                with open(os.path.join('data', 'styles.pkl'), 'rb') as file:
                    styles = pickle.load(file)
        
                if args.style > len(styles[0]):
                    raise ValueError('Requested style is not in style list')
        
                style = [styles[0][args.style], styles[1][args.style]]
        
            phi_data, window_data, kappa_data, stroke_data, coords = sample_text(sess, st, translation, style)
        
            strokes = np.array(stroke_data)
            strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)
            minx, maxx = np.min(strokes[:, 0]), np.max(strokes[:, 0])
            miny, maxy = np.min(strokes[:, 1]), np.max(strokes[:, 1])
        
            if args.info:
                delta = abs(maxx - minx) / 400
                x = np.arange(minx, maxx, delta)
                y = np.arange(miny, maxy, delta)
                x_grid, y_grid = np.meshgrid(x, y)
        
        
                fig, ax = plt.subplots(1,1, figsize = (12,1), dpi = 500,constrained_layout = True)
        
                for stroke in split_strokes(cumsum(np.array(coords))):
                    ax.plot(stroke[:, 0], -stroke[:, 1])
                ax.set_aspect('equal')
        
                plt.axis("off")
                
                coun = coun + 1
                fig.savefig("generated/generated_%s.png" %coun, transparent = True, pad_inches = 0)
                img1 = Image.open(r'generated/generated_%s.png' %coun)
                
            else:
                fig, ax = plt.subplots(1, 1)
                for stroke in split_strokes(cumsum(np.array(coords))):
                    plt.plot(stroke[:, 0], -stroke[:, 1])
                ax.set_title('Handwriting')
                ax.set_aspect('equal')
        
                plt.show();
    
    #Generating the PDF from images in the generated folder
    images_names = []

    for file in os.listdir('generated/'):
    	images_names.append(file)
    
    images_names.sort()
    
    images = [Image.open(f'generated/{img}') for img in images_names]
    
    images = [img.convert('L') for img in images]
    
    images = [img.resize((800,80)) for img in images]
    
    i=0
    
    for img,name in zip(images,images_names):
    	img.save('jpgimages/{}-{}.png'.format(name.split('.')[0],i),'PNG')
    	i+=1
    
    images = [Image.open('jpgimages/{}-{}.png'.format(name.split('.')[0],i)) for i,name in enumerate(images_names)]
    
    new_img_size = (900, 1000)
    
    img_count = 0
    new_image = Image.new('RGB', new_img_size, (255,255,255))
    current_y = 50
    
    new_images_list = []
    
    for img in images:
    	if img_count == 10:
    		new_images_list.append(new_image)
    		new_image = Image.new('RGB', new_img_size, (255,255,255))
    		img_count = 0
    		current_y = 50
    	new_image.paste(img, (50, current_y))
    	img_count+=1
    	current_y+=img.size[1]-10

    if img_count > 0:
    	new_images_list.append(new_image)
    
    img1 = new_images_list[0]
    new_images_list.remove(new_images_list[0])
    img1.save('Handwritten.pdf',save_all=True,append_images=new_images_list)
    
    #Removing temporary files stored in generated folders and jpgimages folder
    '''for root, dirs, files in os.walk('generated'):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))'''

    for root, dirs, files in os.walk('jpgimages'):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

if __name__ == '__main__':
    main()
