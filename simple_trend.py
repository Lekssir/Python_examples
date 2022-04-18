import numpy as np
import tulipy as ti


class Trend:

	def __init__(self, trend_level: int):
		self.price_dir: int = 0
		self.trend_level: int = trend_level
		self.trend = []
		self.trend_e = [[0, 0], [0, 0]]
		for i in range(trend_level + 1):
			self.trend.append([0, 0])

	def stat(self):
		print(self.price_dir)
		for i in range(self.trend_level + 1):
			print(self.trend[i][0], self.trend[i][1])

	@staticmethod
	def hnl(seq):
		return seq[-1] / seq[-2]

	@staticmethod
	def draw_trend2(point1: list, point2: list, point3: list) -> int:
		y = (point3[0] - point1[0]) * (point2[1] - point1[1]) / (point2[0] - point1[0]) + point1[1]
		#plt.plot([point1[0], point2[0], point3[0]], [point1[1], point2[1], y])
		return point3[1] - y

	def update_trend(self, series):
		self.price_dir = (series[-1] - series[-2])
		temp = np.diff(np.sign(np.diff(series)))
		max_arr = (temp < 0).nonzero()[0] + 1
		min_arr = (temp > 0).nonzero()[0] + 1
		#plt.plot(series)
		#plt.plot(max_arr, series[max_arr])#, 'o')
		#plt.plot(min_arr, series[min_arr])#, 'o')
		p1 = 0
		p2 = 0
		if len(max_arr) > 1:
			p1 = [max_arr[-1], series[max_arr[-1]]]
			self.trend_e[0][0] = self.hnl(series[max_arr[-2:]])
			self.trend[0][0] = self.draw_trend2([max_arr[-1], series[max_arr[-1]]], [max_arr[-2], series[max_arr[-2]]],
											[series.size - 1, series[-1]])
		if len(min_arr) > 1:
			p2 = [min_arr[-1], series[min_arr[-1]]]
			self.trend_e[0][1] = self.hnl(series[min_arr[-2:]])
			self.trend[0][1] = self.draw_trend2([min_arr[-1], series[min_arr[-1]]], [min_arr[-2], series[min_arr[-2]]],
											[series.size - 1, series[-1]])

		if self.trend_level > 1:
			series1 = series[max_arr]
			series2 = series[min_arr]
			temp1 = np.diff(np.sign(np.diff(series1)))
			temp2 = np.diff(np.sign(np.diff(series2)))
			max_arr1 = max_arr
			min_arr1 = min_arr
			max_arr = (temp1 < 0).nonzero()[0] + 1
			min_arr = (temp2 > 0).nonzero()[0] + 1
			#plt.plot(max_arr1[max_arr], series1[max_arr])
			#plt.plot(min_arr1[min_arr], series2[min_arr])
			if len(max_arr) > 0:
				self.trend[1][0] = self.draw_trend2(p1, [max_arr1[max_arr[-1]], series1[max_arr[-1]]], [series.size - 1, series[-1]])
			else:
				self.trend[1][0] = self.trend[0][0]
			if len(min_arr) > 0:
				self.trend[1][1] = self.draw_trend2(p2, [min_arr1[min_arr[-1]], series2[min_arr[-1]]], [series.size - 1, series[-1]])
			else:
				self.trend[1][1] = self.trend[0][1]
			if len(max_arr) > 1:
				self.trend_e[1][0] = self.hnl(series[max_arr[-2:]])
				self.trend[2][0] = self.draw_trend2([max_arr1[max_arr[-1]], series1[max_arr[-1]]], [max_arr1[max_arr[-2]], series1[max_arr[-2]]],
											[series.size - 1, series[-1]])
			else:
				self.trend[2][0] = self.trend[0][0]
			if len(min_arr) > 1:
				self.trend_e[1][1] = self.hnl(series[min_arr[-2:]])
				self.trend[2][1] = self.draw_trend2([min_arr1[min_arr[-1]], series2[min_arr[-1]]], [min_arr1[min_arr[-2]], series2[min_arr[-2]]],
											[series.size - 1, series[-1]])
			else:
				self.trend[2][1] = self.trend[0][1]

		for i in range(2, self.trend_level):
			series1 = series1[max_arr]
			series2 = series2[min_arr]
			temp1 = np.diff(np.sign(np.diff(series1)))
			temp2 = np.diff(np.sign(np.diff(series2)))
			max_arr1 = max_arr1[max_arr]
			min_arr1 = min_arr1[min_arr]
			max_arr = (temp1 < 0).nonzero()[0] + 1
			min_arr = (temp2 > 0).nonzero()[0] + 1
			#plt.plot(max_arr1[max_arr], series1[max_arr])
			#plt.plot(min_arr1[min_arr], series2[min_arr])
			if len(max_arr) > 1:
				self.trend[1][0] = self.draw_trend2([max_arr1[max_arr[-1]], series1[max_arr[-1]]], [max_arr1[max_arr[-2]], series1[max_arr[-2]]],
											[series.size - 1, series[-1]])
			else:
				self.trend[1][0] = 0
			self.trend[1][1] = self.draw_trend2([min_arr1[min_arr[-1]], series2[min_arr[-1]]], [min_arr1[min_arr[-2]], series2[min_arr[-2]]],
											[series.size - 1, series[-1]])
		#self.stat()
		#plt.show()

	@staticmethod
	def calc_weight(data):
		print(data)
		trend = Trend(2)
		trend.update_trend(data)

		trend_rsi = Trend(2)
		rsi_data = ti.rsi(data, 14)
		trend_rsi.update_trend(rsi_data)

		weight_price = analyse_trend(trend.trend, trend.price_dir)
		print(trend.trend)
		#weight_rsi = analyse_trend(trend_rsi.trend, trend_rsi.price_dir)
		#res = weight_rsi #* 1 + weight_price * 0
		res = weight_price
		#res = self.analizator01(trend.trend, trend.trend_e, trend_rsi.trend, trend_rsi.trend_e)
		return res

	@staticmethod
	def analizator01(price_c, price_e, rsi_c, rsi_e):
		tr0 = (price_e[0][0] + price_e[0][1]) / 2
		tr1 = (price_e[1][0] + price_e[1][1]) / 2
		tri0 = (rsi_e[0][0] + rsi_e[0][1]) / 2
		tri1 = (rsi_e[1][0] + rsi_e[1][1]) / 2

		#return ((tr1 - 1) + (tr0 - 1) + (tri0 - 1) + (tri0 - 1)) / 4
		if np.sign(tr0 - 1) == np.sign(tr1 - 1):
			tr = tr1 - 1
		else:
			tr = 0
		if np.sign(tri0 - 1) == np.sign(tri1 - 1):
			tri = tri1 - 1
		else:
			tri = 0
		return tri0 - 1

	@staticmethod
	def weighted_mean(seq, weights):
		res = 0
		for i in range(len(seq)):
			res += seq[i] * weights[i]
		return res / len(seq)

	@staticmethod
	def rsi_analyze(trend_lines, price_direction):
		####TODO
		return 0

	@staticmethod
	def analyse_trend(trend_lines, price_direction):
		temp = [0, 0, 0]
		eps = 0.1
		for i in range(len(trend_lines)):
			if price_direction >= 0:
				if trend_lines[i][0] > eps:
					temp[i] = 1
				elif trend_lines[i][0] > (-1 * eps):
					temp[i] = 0
				elif trend_lines[i][1] > eps:
					temp[i] = 1
				else:
					temp[i] = 0
			else:
				if trend_lines[i][0] > (-1 * eps):
					temp[i] = 0
				elif trend_lines[i][1] > eps:
					temp[i] = -1
				elif trend_lines[i][1] > (-1 * eps):
					temp[i] = 0
				else:
					temp[i] = -1
		if temp[0] == temp[1] == temp[2]:
			return temp[0]
		else:
			return 0
