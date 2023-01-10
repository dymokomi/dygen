from PIL import Image, ImageFilter, ImageStat
import random
from tqdm import tqdm
class DYImage:
    def __init__(self, path="", img=None):
        if img == None and path != "":
            self.__pilimage = Image.open(path).convert("RGB")
        elif img != None:
            self.__pilimage = img.convert("RGB")
        else:
            self.__pilimage = None

    def pil(self):
        return self.__pilimage

    def show(self):
        self.__pilimage.show()

    def size(self):
        return self.__pilimage.size

    def quantized(self, colors=16, method=0, kmeans=1):
        return DYImage(img = self.__pilimage.quantize(colors=colors, method=method, kmeans=kmeans).convert('RGB'))
    def rgb(self):
        return DYImage(img = self.__pilimage.convert('RGB'))

    def expanded(self, size):
        new_img = self.__pilimage.copy()
        if size[0] > new_img.size[0]:
            temp_image = Image.new(mode='RGB', size=(size[0], new_img.size[1]))
            for i in range(int (size[0] / new_img.size[0])+1):
                Image.Image.paste(temp_image, new_img, (new_img.size[0]*i,0))

            new_img = temp_image 

        if size[1] > new_img.size[1]:
            temp_image = Image.new(mode='RGB', size=(new_img.size[0],size[1]))
            for i in range(int (size[1] / new_img.size[1])+1):
                Image.Image.paste(temp_image, new_img, (0, new_img.size[1]*i))

            new_img = temp_image 
        return(DYImage(img = new_img))

    def pixilated(self, size=16, offset=0, prob=1.0, checker=False):

        new_img = self.__pilimage.copy()
        source_pixels = self.__pilimage.load()
        pixels = new_img.load()
        pcoord = {}
        for x in range(new_img.size[0]):
            for y in range(new_img.size[1]):
                if type(size) is tuple:
                    if checker:
                        if size[0] <= size[1]:
                            if ((((x) // size[0])) % 2) == 0:
                                vertical_offset =  (size[1]//2)
                            else :
                                vertical_offset = 0
                            coord =  (( (x) // size[0]) * size[0], ((y+vertical_offset+offset)// size[1]) * size[1] - vertical_offset-offset )
                        else:
                            if ((((y) // size[1])) % 2) == 0:
                                horizontal_offset =  (size[0]//2)
                            else :
                                horizontal_offset = 0
                            coord =  (( (x+horizontal_offset) // size[0]) * size[0] - horizontal_offset, ((y+offset)// size[1]) * size[1]-offset )
                    else:
                        coord =  (( (x+offset) // size[0]) * size[0] - offset, ((y+offset)// size[1]) * size[1] -offset)
                else:
                    coord =  (( (x) // size) * size, ((y+offset) // size) * size -offset)

                if coord in pcoord.keys():
                    if pcoord[coord]:
                        pixels[x,y] = source_pixels[ coord[0],coord[1]]
                else:
                    pcoord[coord] = random.random() <= prob
                    if pcoord[coord]:
                        pixels[x,y] = source_pixels[ coord[0],coord[1]]

        return DYImage(img = new_img)

    def palette(self):
        colors = self.__pilimage.getcolors()
        return [x[1] for x in colors]

        pixels = self.__pilimage.load()
        colors = []
        
        for x in range(self.__pilimage.size[0]):
            for y in range(self.__pilimage.size[1]):
                colors.append(pixels[x,y])
        return list(set)
        

    def doubled(self):
        new_img = self.__pilimage.copy()
        return DYImage(img = new_img).scaled(2)
    def resized(self, size):
        self.__pilimage = self.__pilimage.resize( size, Image.Resampling.BICUBIC) 
        return self
    def scaled(self, mult=2):
        self.__pilimage = self.__pilimage.resize( (self.__pilimage.size[0] * mult,self.__pilimage.size[1] * mult), Image.Resampling.NEAREST) 
        return self

    def extract(self, palette=None):
        if palette == None:
            palette = self.palette()

        masks = []
        source_pixels = self.__pilimage.load()
        for color in palette:
            img = Image.new("L", self.__pilimage.size)
            pixels = img.load()
            for x in range(img.size[0]):
                for y in range(img.size[1]): 
                    if source_pixels[x,y] == color:
                        pixels[x,y] = 255
                    else:
                        pixels[x,y] = 0
            masks.append(DYImage(img = img))
        return masks  

    def blur(self, radius=4):
        return(DYImage(img = self.__pilimage.filter(ImageFilter.GaussianBlur(radius = radius)))) 

    def clamp(self, clamp=128):
        new_img = self.__pilimage.copy()
        source_pixels = self.__pilimage.load()
        pixels = new_img.load()
        for x in range(new_img.size[0]):
            for y in range(new_img.size[1]):
                if source_pixels[x,y][0] > clamp:
                    pixels[x,y] = (255,255,255)
                else:
                    pixels[x,y] = (0,0,0)
        return DYImage(img = new_img)

    def rounded(self, radius=8, clamp=128):
        return self.blur(radius=radius).clamp(clamp)

    def cleanup(self, radius=13):
        new_img = self.__pilimage.copy()
        new_img = new_img.filter(ImageFilter.ModeFilter(size=radius))
        
        return(DYImage(img = new_img)) 
    def cleanup2(self, radius=13):
        new_img = self.__pilimage.copy()
        new_img = new_img.filter(ImageFilter.MedianFilter(size=radius))
        
        return(DYImage(img = new_img)) 

    def cleanup_map(self, map, radius_a = 3, radius_b = 17):
        levels = (radius_b - radius_a+1)//2
        quants = []
        for i in tqdm(range(levels)):
            cp = self.__pilimage.copy()
            quants.append( cp.filter(ImageFilter.ModeFilter(size=radius_a + (i*2))).load() )

        quantized_map = map.resized(self.__pilimage.size).pil().quantize(colors=levels, method=0, kmeans=1)
        
        new_img = self.__pilimage.copy()
        pixels = new_img.load()
        map_pixels = quantized_map.load()
        for x in range(new_img.size[0]):
            for y in range(new_img.size[1]): 
                ix = map_pixels[x,y]
                pixels[x,y] = quants[ix][x,y]
        return DYImage(img = new_img.convert("RGB"))

    def pixels(self):
        return self.__pilimage.load()
    def averaged(self):
        mean_color = ImageStat.Stat(self.__pilimage).mean
        mean_color = (int(mean_color[0]), int(mean_color[1]),int(mean_color[2]))
        new_img = Image.new("RGB", self.__pilimage.size, mean_color)
        return(DYImage(img=new_img))
    def multed(self, mask, threshold=True):
        new_img = self.__pilimage.copy()
        mask_pixels = mask.pixels()
        pixels = new_img.load()
        for x in range(new_img.size[0]):
            for y in range(new_img.size[1]): 
                if not threshold:
                    pixels[x,y] = (
                        int(pixels[x,y][0] * (mask_pixels[x,y][0]/255.0)),
                        int(pixels[x,y][1] * (mask_pixels[x,y][1]/255.0)),
                        int(pixels[x,y][2] * (mask_pixels[x,y][2]/255.0))
                    )
                else:
                    if mask_pixels[x,y][0] < 128:
                        pixels[x,y] = (0,0,0)
        return DYImage(img = new_img)
    def mult(self, amount= 1.0):
        new_img = self.__pilimage.copy()
        pixels = new_img.load()
        for x in range(new_img.size[0]):
            for y in range(new_img.size[1]): 
                pixels[x,y] = ( min(255,int(pixels[x,y][0]*amount)),
                                min(255,int(pixels[x,y][1]*amount)),
                                min(255,int(pixels[x,y][2]*amount))
                                )
        return DYImage(img = new_img)
    def cleared(self, color=(255,255,255)):
        new_img = self.__pilimage.copy()
        pixels = new_img.load()
        for x in range(new_img.size[0]):
            for y in range(new_img.size[1]): 
                pixels[x,y] = color
        return DYImage(img = new_img)
    def blend(self, target, amount=0.5):
        pixels = self.__pilimage.load()
        target_pixels = target.pixels()
        for x in range(self.__pilimage.size[0]):
            for y in range(self.__pilimage.size[1]):
                pixels[x,y] = ( int(pixels[x,y][0]*(1.0-amount) + target_pixels[x,y][0]*(amount)), 
                                int(pixels[x,y][1]*(1.0-amount) + target_pixels[x,y][1]*(amount)),
                                int(pixels[x,y][2]*(1.0-amount) + target_pixels[x,y][2]*(amount)))
    def paint(self, mask, color, spread=0, amount=1.0, stencil=None, threshold=True, volume_diff=1.0, only_add=False):
        pixels = self.__pilimage.load()
        mask_pixels = mask.pixels()
        stencil_pixels = None if stencil == None else stencil.pixels()
        rand_v = random.randint(-3,3)
        color = (
            min(255, max(0, color[0] + rand_v)), 
            min(255, max(0, color[1] + rand_v)),
            min(255, max(0, color[2] + rand_v))
            )
        for x in range(self.__pilimage.size[0]):
            for y in range(self.__pilimage.size[1]):
                if not threshold:
                    if volume_diff>= 1.0:
                        pixels[x,y] = ( 
                                        int( pixels[x,y][0]*(1.0-amount) + (pixels[x,y][0]*(1.0-mask_pixels[x,y][0]/255.0) + (color[0]*(mask_pixels[x,y][0]/255.0)))*amount), 
                                        int( pixels[x,y][1]*(1.0-amount) + (pixels[x,y][1]*(1.0-mask_pixels[x,y][0]/255.0) + (color[1]*(mask_pixels[x,y][1]/255.0)))*amount), 
                                        int( pixels[x,y][2]*(1.0-amount) + (pixels[x,y][2]*(1.0-mask_pixels[x,y][0]/255.0) + (color[2]*(mask_pixels[x,y][2]/255.0)))*amount)
                        )
                    else:
                        temp_pix = ( 
                                        int( pixels[x,y][0]*(1.0-amount) + (pixels[x,y][0]*(1.0-mask_pixels[x,y][0]/255.0) + (color[0]*(mask_pixels[x,y][0]/255.0)))*amount), 
                                        int( pixels[x,y][1]*(1.0-amount) + (pixels[x,y][1]*(1.0-mask_pixels[x,y][0]/255.0) + (color[1]*(mask_pixels[x,y][1]/255.0)))*amount), 
                                        int( pixels[x,y][2]*(1.0-amount) + (pixels[x,y][2]*(1.0-mask_pixels[x,y][0]/255.0) + (color[2]*(mask_pixels[x,y][2]/255.0)))*amount)
                        )
                        tv = (temp_pix[0]/255.0 + temp_pix[1]/255.0 + temp_pix[2]/255.0) / 3.0
                        pv = (pixels[x,y][0]/255.0 + pixels[x,y][1]/255.0 + pixels[x,y][2]/255.0) / 3.0
                        if abs(pv-tv) < volume_diff:
                            pixels[x,y] = temp_pix
                else:
                    if spread > 0:
                        kx = x + random.randint(-spread, spread)
                        ky = y + random.randint(-spread, spread)
                        if kx < 0: kx = 0
                        if ky < 0: ky = 0
                        if kx >= self.__pilimage.size[0]: kx = self.__pilimage.size[0] -1 
                        if ky >= self.__pilimage.size[1]: ky = self.__pilimage.size[1] -1 
                        if mask_pixels[kx,ky][0] > 128 and (True if stencil==None else (stencil_pixels[x,y][0])):
                            if only_add:
                                if amount == 1.0 and (pixels[x,y][0]<color[0] and pixels[x,y][1]<color[1] and pixels[x,y][2]<color[2]): 
                                    pixels[x,y] = color 
                                elif (pixels[x,y][0]<color[0] and pixels[x,y][1]<color[1] and pixels[x,y][2]<color[2]):
                                    pixels[x,y] = ( 
                                                        int(pixels[x,y][0]*(1.0-amount) + (color[0]*amount)), 
                                                        int(pixels[x,y][1]*(1.0-amount) + (color[1]*amount)), 
                                                        int(pixels[x,y][2]*(1.0-amount) + (color[2]*amount))
                                    )
                            else:
                                if amount == 1.0: pixels[x,y] = color 
                                else: pixels[x,y] = ( 
                                                        int(pixels[x,y][0]*(1.0-amount) + (color[0]*amount)), 
                                                        int(pixels[x,y][1]*(1.0-amount) + (color[1]*amount)), 
                                                        int(pixels[x,y][2]*(1.0-amount) + (color[2]*amount))
                                )
                    else:
                        if mask_pixels[x,y][0] > 128:
                            if amount == 1.0: pixels[x,y] = color 
                            else: pixels[x,y] = ( 
                                                    int(pixels[x,y][0]*(1.0-amount) + (color[0]*amount)), 
                                                    int(pixels[x,y][1]*(1.0-amount) + (color[1]*amount)), 
                                                    int(pixels[x,y][2]*(1.0-amount) + (color[2]*amount))
                            )

    def comp(self, target, mask):
        pixels = self.__pilimage.load()
        target_pixels = target.pixels()
        mask_pixels = mask.pixels()
        for x in range(self.__pilimage.size[0]):
            for y in range(self.__pilimage.size[1]):
                if mask_pixels[x,y][0] > 128:
                    pixels[x,y] = target_pixels[x,y] 
    def darkened_edges(self, amount=0.5):
        dark_edges = self.__pilimage.filter(ImageFilter.FIND_EDGES).convert("L")
        new_img = self.__pilimage.copy()
        pixels = new_img.load()
        edges_pixels = dark_edges.load()
        for x in range(new_img.size[0]):
            for y in range(new_img.size[1]): 
                pixels[x,y] = (
                    int(pixels[x,y][0] - edges_pixels[x,y]*amount),
                    int(pixels[x,y][1] - edges_pixels[x,y]*amount),
                    int(pixels[x,y][2] - edges_pixels[x,y]*amount)
                )
        return DYImage(img = new_img)
    def fringe(self, r=5, g=0, b=-3):
    
        new_img = self.__pilimage.copy()
        pixels = new_img.load()
        source_pixels = self.__pilimage.load()

        center_x = self.__pilimage.size[0] // 2
        center_y = self.__pilimage.size[1] // 2
        for x in range(new_img.size[0]):
            for y in range(new_img.size[1]): 
                ditance_x = abs(center_x - x) / center_x
                ditance_y = abs(center_y - y) / center_y

                tx_r =  max(0, min(self.__pilimage.size[0]-1, x + int(ditance_x * r)))
                ty_r =  max(0, min(self.__pilimage.size[1]-1, y + int(ditance_y * r)))

                tx_g =  max(0, min(self.__pilimage.size[0]-1, x + int(ditance_x * g)))
                ty_g =  max(0, min(self.__pilimage.size[1]-1, y + int(ditance_y * g)))

                tx_b =  max(0, min(self.__pilimage.size[0]-1, x + int(ditance_x * b)))
                ty_b =  max(0, min(self.__pilimage.size[1]-1, y + int(ditance_y * b)))

                pixels[x,y] = (
                    source_pixels[tx_r, ty_r][0],
                    source_pixels[tx_g, ty_g][1],
                    source_pixels[tx_b, ty_b][2]
                )
        return DYImage(img = new_img)
    def offset(self, r=0, g=14, b=18):
    
        new_img = self.__pilimage.copy()
        pixels = new_img.load()
        source_pixels = self.__pilimage.load()
        for x in range(new_img.size[0]):
            for y in range(new_img.size[1]): 
                
                pixels[x,y] = (
                    source_pixels[x,y][0] + r,
                    source_pixels[x,y][1] + g,
                    source_pixels[x,y][2] + b,
                )
        return DYImage(img = new_img)
    def glow(self, threshold, radius, amount):
        glow_img = self.__pilimage.copy()
        glow_pixels = glow_img.load()
        for x in range(glow_img.size[0]):
            for y in range(glow_img.size[1]): 
                if (glow_pixels[x,y][0] / 255.0 + glow_pixels[x,y][1] / 255.0 + glow_pixels[x,y][2] / 255.0)/3.0 < threshold:
                    glow_pixels[x,y] = (0,0,0)
        glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius = radius))
        glow_pixels = glow_img.load()

        new_image = self.__pilimage.copy()
        pixels = new_image.load()
        for x in range(new_image.size[0]):
            for y in range(new_image.size[1]): 
                pixels[x,y] = (min(255, pixels[x,y][0] + int(glow_pixels[x,y][0] * amount[0])),
                        min(255, pixels[x,y][1] + int(glow_pixels[x,y][1] * amount[1])),
                        min(255, pixels[x,y][2] + int(glow_pixels[x,y][2] * amount[2]))
                    )
        return DYImage(img = new_image)
    def glow_blur(self, threshold, radius, amount):
        glow_img = self.__pilimage.copy()
        glow_pixels = glow_img.load()
        for x in range(glow_img.size[0]):
            for y in range(glow_img.size[1]): 
                if (glow_pixels[x,y][0] / 255.0 + glow_pixels[x,y][1] / 255.0 + glow_pixels[x,y][2] / 255.0)/3.0 < threshold:
                    glow_pixels[x,y] = (0,0,0)
        glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius = radius))
        glow_pixels = glow_img.load()

        new_image = self.__pilimage.copy()
        pixels = new_image.load()
        for x in range(new_image.size[0]):
            for y in range(new_image.size[1]): 
                pixels[x,y] = (min(255, int(pixels[x,y][0]*(1.0-amount[0]) + glow_pixels[x,y][0] * amount[0])),
                               min(255, int(pixels[x,y][1]*(1.0-amount[1]) + glow_pixels[x,y][1] * amount[1])),
                               min(255, int(pixels[x,y][2]*(1.0-amount[2]) + glow_pixels[x,y][2] * amount[2]))
                    )
        return DYImage(img = new_image)
    def sharpen(self):
        new_img = self.__pilimage.copy()
        new_img = new_img.filter(ImageFilter.SHARPEN)
        return DYImage(img = new_img)
    def save(self, path, mult=1):
        if mult == 1:
            self.__pilimage.save(path)
        else:
            img_copy = self.__pilimage.copy()
            img_copy = img_copy.resize( (img_copy.size[0]//mult, img_copy.size[1]//mult), Image.Resampling.NEAREST)
            img_copy.save(path)
    def load(self, path, mult=1):
        self.__pilimage = Image.open(path)
        if mult != 1:
            self.__pilimage = self.__pilimage.resize( (self.__pilimage.size[0] * mult, self.__pilimage.size[1] * mult), Image.Resampling.NEAREST )