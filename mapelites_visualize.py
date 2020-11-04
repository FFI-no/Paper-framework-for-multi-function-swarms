import os
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as plticker
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Circle

from subprocess import call
from mapelites_train import Genome
from ndimarray import LabeledNDimArray
from utilities import open_pickle, Translator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

class RepertoireFigure(object):
	def __init__(self, output_folder, filetype="pdf"):
		self._output_folder = output_folder
		self._figure_subfolder = ""
		self._filetype = filetype
	
	def parse_checkpoint(self, name, chkpt):
		b = BehaviorSpace(os.path.join(self._output_folder, self._figure_subfolder, name), chkpt)
		b.visualize(True, self._filetype, False)

	def start_experiment(self, name=""):
		self._figure_subfolder = name
		if not os.path.exists(os.path.join(self._output_folder, self._figure_subfolder)):
			os.makedirs(os.path.join(self._output_folder, self._figure_subfolder))

	def end_experiment(self, name=""):
		return

	def all_done(self):
		return

class BehaviorSpace(object):
	def __init__(self, name, mapelites, diff_mapelites=None):
		self._name = name

		pairs = [("exploration_median", "Exploration"), ("localization_variance", "Localization variance"), ("network_covered", "Network coverage")]
		self._trans = Translator(pairs)

		self._cmap = self._generate_colormap()

		solutions = mapelites.get_all_solutions()
		print len(solutions)
		self._intertia = None# self._calc_intertia(solutions)
		original_labels = solutions[0][-1].characteristics.keys()
		self._ndm = LabeledNDimArray(mapelites._dims, solutions, labels=original_labels)

		self._f, self._axarr = plt.subplots(11, sharex=True, sharey=True, figsize=(8,8))
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.04)

		#self._f.suptitle("MAPElites repertoire: %s" %name)

		#self._axarr[0].text(-0.3, 1.3, self._trans[original_labels[1]], transform=self._axarr[0].transAxes, size=14)

		self._draw_axis_labels(original_labels)

	def _draw_axis_labels(self, original_labels):

		plt.gca().annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), 
            arrowprops=dict(arrowstyle="<-", color='black'))
		plt.gca().annotate(self._trans[original_labels[2]], xy=(0, -0.4), xycoords='axes fraction')

		plt.gca().annotate('', xy=(-0.01, 0), xycoords='axes fraction', xytext=(-0.01, 1), 
            arrowprops=dict(arrowstyle="<-", color='black'))
		plt.gca().annotate(self._trans[original_labels[0]], xy=(-0.05, 1.25), xycoords='axes fraction', rotation=90)

		plt.gca().annotate('', xy=(-0.065, 0), xycoords='axes fraction', xytext=(-0.065, 11.4), 
            arrowprops=dict(arrowstyle="<-", color='black'))
		plt.gca().annotate(self._trans[original_labels[1]], xy=(-0.105, 2.5), xycoords='axes fraction', rotation=90)
		#plt.gca().annotate(self._trans[original_labels[2]], xy=(0, -0.4), xycoords='axes fraction', rotation=-90)

	def _draw_selected_behaviors(self):
		pos_list = []

		pos_list.append((0,1,20))
		pos_list.append((0,9,20))
		pos_list.append((0,1,59))
		pos_list.append((0,9,40))
		pos_list.append((0,3,19))
		pos_list.append((0,3,60))
		pos_list.append((0,3,97))
		pos_list.append((4,3,19))
		pos_list.append((5,3,60))
		pos_list.append((4,3,94))
		pos_list.append((0,6,19))
		pos_list.append((0,6,60))
		pos_list.append((0,6,98))
		pos_list.append((4,6,19))
		pos_list.append((6,6,60))
		pos_list.append((4,6,92))

		base = plt.cm.get_cmap("jet")
		clist = base(np.linspace(0,1, 16))
		cmap = base.from_list("Sequence", clist, 16)

		for ix, pos in enumerate(pos_list):
			i = 10-pos[0]
			j = 10-pos[1]
			k = pos[2]-15

			p = Circle((k,i), radius=1, edgecolor="r",linewidth=1.5,fill=False)

			self._axarr[j].add_artist(p)

	def _create_color_map_selected(self):
		base = plt.cm.get_cmap("jet")
		clist = base(np.linspace(0,1, 16))
		cmap = base.from_list("Sequence", clist, 16)

		return cmap

	def _create_translator_acr_index(self):
		translation = {}

		translation[(0,1,20)] = "LLL"
		translation[(0,9,20)] = "LLH"
		translation[(0,1,59)] = "LHL"
		translation[(0,9,40)] = "LHH"
		translation[(0,3,19)] = "LL4"
		translation[(0,3,60)] = "LM4"
		translation[(0,3,97)] = "LH4"
		translation[(4,3,19)] = "HL4"
		translation[(5,3,60)] = "HM4"
		translation[(4,3,94)] = "HH4"
		translation[(0,6,19)] = "LL7"
		translation[(0,6,60)] = "LM7"
		translation[(0,6,98)] = "LH7"
		translation[(4,6,19)] = "HL7"
		translation[(6,6,60)] = "HM7"
		translation[(4,6,92)] = "HH7"

		translation_r = {}
		for key, value in translation.items():
			translation_r[value] = key

		return translation, translation_r

	def _mapelites_to_matplot_coords(self, pos):
		new_pos = [0.,0.,0.]
		new_pos[0] = pos[2]-15
		new_pos[1] = 10-pos[0]
		new_pos[2] = 10-pos[1]
		return new_pos
		
	def _draw_uncertainty_ellipses(self):
		ellipse_list = []

		ellipse_list.append([(0, 0, 0), 'LLL', 0.029111111111111112, 0.049858565394622779, 0.21361458429517785, 0.016200190753385853, 0.23913325744727551, 0.083548092176796748])
		ellipse_list.append([(0, 0, 0), 'LLH', 0.07555555555555557, 0.060614140875492682, 0.24618008964784915, 0.068420485555527946, 0.84116194514748255, 0.12278901947424353])
		ellipse_list.append([(0, 0, 0), 'LHL', 0.094000000000000028, 0.045580480985983239, 0.64000575324994136, 0.10597285271247535, 0.2464153819298818, 0.054406335285651475])
		ellipse_list.append([(0, 0, 0), 'LHH', 0.074444444444444452, 0.060563200332940645, 0.35450107164966305, 0.12815877103471268, 0.75808545603646571, 0.16524558992501973])
	
		ellipse_list.append([(0, 0, 0), 'HH7', 0.47388888888888897, 0.0533767184648333, 0.82298355806511747, 0.10757305938037866, 0.54234853000653971, 0.1219336076320712])
		ellipse_list.append([(0, 0, 0), 'LH7', 0.08100000000000003, 0.050130570254826277, 0.86603269463363064, 0.19775545536829056, 0.67232741403193252, 0.040826380512572431])
		ellipse_list.append([(0, 0, 0), 'HL7', 0.35944444444444446, 0.092012814459135839, 0.20347661975903608, 0.01479413356007446, 0.51186654070212501, 0.086110391230410768])
		ellipse_list.append([(0, 0, 0), 'LL4', 0.12333333333333338, 0.056426506335976095, 0.20177677610076419, 0.0062490503072605936, 0.41835895552523267, 0.067847210904788999])
		ellipse_list.append([(0, 0, 0), 'HL4', 0.39755555555555555, 0.0672122122942065, 0.24191469993881606, 0.053528552358672772, 0.39752835543905712, 0.053868629874039134])
		ellipse_list.append([(0, 0, 0), 'HM4', 0.50800000000000001, 0.14387992250055404, 0.61102390520259986, 0.095005346048143702, 0.48461553282980235, 0.11536637075180142])
		ellipse_list.append([(0, 0, 0), 'HH4', 0.47211111111111115, 0.066201991689855577, 0.87518101816214122, 0.13199021734355745, 0.44672964006320259, 0.066871421964889183])
		ellipse_list.append([(0, 0, 0), 'LL7', 0.12100000000000002, 0.058275526913422554, 0.20117962461239064, 0.0092269270429343481, 0.60183284423842154, 0.079460780311181645])
		ellipse_list.append([(0, 0, 0), 'LM7', 0.014, 0.036872151490732891, 0.53965426722791876, 0.11533598451574793, 0.5924700494764068, 0.1088570325341377])
		ellipse_list.append([(0, 0, 0), 'LM4', 0.033000000000000002, 0.050770726473694135, 0.5120736354603016, 0.16893565205769109, 0.42496613547667383, 0.097444977875356928])
		ellipse_list.append([(0, 0, 0), 'LH4', 0.11011111111111115, 0.04229803135832829, 0.88541968848081243, 0.095597200346456759, 0.39570946439956056, 0.047540695194206471])
		ellipse_list.append([(0, 0, 0), 'HM7', 0.56644444444444442, 0.096506636373865667, 0.55107752421562772, 0.07929758452162719, 0.57448997202392815, 0.12917864901777704])
		
		translation, translation_r = self._create_translator_acr_index()

		colors = ["#91BE27", "#91BE27", "#91BE27", "#5F2387","#91BE27","#F33442","#000000","#000000","#F33442","#5F2387","#5F2387","#5F2387","#91BE27","#000000","#000000","#F33442"]

		for ix,ellipse in enumerate(ellipse_list):

			_,name,expl_mean, expl_std, net_mean, net_std, loc_mean, loc_std = ellipse

			print ellipse

			indexes = translation_r[name]
			count = 0			
			for axi in range(10):
				if abs(axi*0.1 - loc_mean)/(2*loc_std) > 1.:
					continue

				z_offset = axi * 0.1 - loc_mean
				azimuth = 3.14/2
				azimuth = np.arccos(z_offset/(2*loc_std))

				pos = [expl_mean*10., loc_mean*10., net_mean*100.]

				pos = [indexes[0], max(0, min(9,int(loc_mean/0.1+0.00001))), indexes[2]]
				pos = [indexes[0], axi, indexes[2]]

			 	matplotlib_pos = self._mapelites_to_matplot_coords(pos)

				ellipse = Ellipse(xy=(matplotlib_pos[0],matplotlib_pos[1]), width=2.*net_std * np.sin(azimuth)*100., height=2.*expl_std * np.sin(azimuth)*10., 
				                        edgecolor=colors[ix], fc='None', lw=2)

				self._axarr[matplotlib_pos[2]].add_patch(ellipse)
				cont_params = self._ndm[int(indexes[0]),int(indexes[1]),int(indexes[2])]

				count += 1

			pos = translation_r[name]

			i = 10-pos[0]
			j = 10-pos[1]
			k = pos[2]-15

			p = Circle((k,i), radius=1, edgecolor=colors[ix],linewidth=1.5,fill=False)

			self._axarr[j].add_artist(p)

	def visualize(self, save_figure=False, filetype="png", show_figure=True):
		def calc_median(matrix2d):
			sequence = []
			for row in matrix2d:
				for value in row:
					if value is not None:
						sequence.append(value)
			sequence = sorted(sequence)
			if len(sequence) == 0:
				return 0
			mi = len(sequence)/2

			if mi%2 == 0:
				return (sequence[mi]+sequence[mi-1])/2
			else:
				return sequence[mi]

		for i in range(11):
			array2d = self._ndm.reduce((None,i,None))
			labels = array2d.get_labels()
			matrix2d_fitness = array2d.get_2d_array(fill=-1., filter_=lambda x: 2.*x.fitness)

			cax = self._plot_map_slice(self._axarr[10-i], i, labels, matrix2d_fitness)

		#self._set_labels(self._axarr, labels)

		self._draw_selected_behaviors()
		#self._draw_uncertainty_ellipses()

		#Draw colorbar
		# norm = colors.Normalize(vmin=0,vmax=1.68)
		# cmap = self._generate_colormap()
		# mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
		# mappable.set_array([])
		# plt.colorbar(mappable, ax=self._axarr.ravel().tolist())

		if save_figure:
			plt.savefig("%s.%s" % (self._name,filetype))

		if show_figure:
			plt.ioff()
			cid = self._f.canvas.mpl_connect('button_press_event', self._onclick)
			plt.show()

		plt.close()

	def _onclick(self, event):
		print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
			  ('double' if event.dblclick else 'single', event.button,
			   event.x, event.y, event.xdata, event.ydata))
		indexes =  map(int, [ event.xdata, event.ydata])
		indexes.reverse()

		i = 10-indexes[0]
		k = indexes[1]+15

		j = 0
		for x, ax in enumerate(self._axarr):
			if ax == event.inaxes:
				j = x
				break
		j = 10-j

		cont_params = self._ndm[i,j,k]
		print (i, j, k), cont_params, cont_params.characteristics
		cont_type = "parametric"
		call(["python", "mapelites_testcontroller.py", "--duration", "1600", cont_type, str(cont_params)])

	def _set_labels(self, axarr, labels):
		loc = plticker.MultipleLocator(base=5.0) 
		plt.gca().yaxis.set_major_locator(loc)
		loc = plticker.MultipleLocator(base=1.0)
		plt.gca().xaxis.set_minor_locator(loc)
		loc = plticker.MultipleLocator(base=1.0) 
		plt.gca().yaxis.set_minor_locator(loc)

		#plt.setp( axarr[-1].get_xticklabels(), visible=True)
		#plt.gca().yaxis.set_ticks_position('left')

		#plt.xlabel(self._trans[labels[1]])
		#plt.gca().set_ylabel(self._trans[labels[0]], ha='left', y=1.)

	def _generate_colormap(self):
		startcolor = '#000000'
		midcolor = '#FFFFFF'
		endcolor = '#0000FF'
		cmap = LinearSegmentedColormap.from_list( 'own2', [startcolor, midcolor, endcolor] )

		colors1 = [np.array([1,1,1,1.])]*128 
		colors2 = plt.cm.viridis(np.linspace(0, 1, 128))

		colors = np.vstack((colors1, colors2))
		cmap = LinearSegmentedColormap.from_list('my_colormap', colors)

		return cmap

	def _plot_map_slice(self, ax, i, labels, matrix2d_fitness):
		ma = np.flipud(np.matrix(matrix2d_fitness)[:, [x for x in range(15,100)]])

		#Use to generate ellipse plot
		#cax = ax.matshow(ma,  vmin=-1., vmax=1.68, cmap=self._cmap, alpha=0.3)

		cax = ax.matshow(ma,  vmin=-1., vmax=1.68, cmap=self._cmap, alpha=1.)

		ax.yaxis.set_ticks_position('none')
		ax.xaxis.set_ticks_position('none')
		plt.setp( ax.get_yticklabels(), visible=False)
		plt.setp( ax.get_xticklabels(), visible=False)

		return cax

def create_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument("logname",  type=str, help="Can be filename or folder")
	parser.add_argument("--difflog",  type=str, default=None)

	parser.add_argument('--pdf', dest='filetype', action='store_const', const="pdf")
	parser.set_defaults(filetype="png")
	return parser

def visualize_one(visualizer, logname, difflogname, filetype):
	figure_name = logname.split("/")[-1]
	b = visualizer(figure_name, open_pickle(logname), diff_mapelites=open_pickle(difflogname))
	b.visualize(filetype=filetype, save_figure=True)

def visualize_many(visualizer, logfolder, filetype):
	for filename in os.listdir(logfolder):
		name, ext = filename.split(".")
		if ext == "chkpt":
			figure_name = filename.split("/")[-1]
			logfilename = os.path.join(logfolder, filename)
			b = visualizer(figure_name, open_pickle(logfilename))
			b.visualize(filetype=args.filetype, save_figure=True, show_figure=False)
			plt.close()

if __name__ == '__main__':
	parser = create_parser()
	args = parser.parse_args()

	visualizer = BehaviorSpace

	if os.path.isfile(args.logname):
		visualize_one(visualizer, args.logname, args.difflog, args.filetype)
	else:
		visualize_many(visualizer, args.logname, args.filetype)
