# -*- coding: utf-8 -*-
"""
@Project:   DynTexTrans
@File   :   vis
@Author :   TonyMao@AILab
@Date   :   2019-08-09
@Desc   :   None
"""

import os

"""
	<tr>
		<td>0</td>
		<td><img width="128" src="result/gt.png"></td>
		<td><img width="128" src="result/gt.png"></td>
		<td><img width="128" src="result/gt.png"></td>
	</tr>
"""


class Table(object):
    def __init__(self):
        self.img_list = {}
        self.header = """
<html class="gr__10_224_7_179"><head>
</head>

<body data-gr-c-s-loaded="true">
<style>* {font-size: 24px;}</style><table>
<tbody>
	<tr>
		<td>index</td>
		<td>Tt</td>
		<td>Tt+1</td>
		<td>PrTt</td>
		<td>PrTt+1</td>
	</tr>
"""
        self.tail = """
</tbody></table>
</body></html>
        """

        self.format = """
    <tr>
		<td>{}</td>
		<td><img src="{}"></td>
		<td><img src="{}"></td>
		<td><img src="{}"></td>
		<td><img src="{}"></td>
	</tr>"""

    def add(self, key: str, url: str):
        if not key in self.img_list.keys():
            self.img_list[key] = [url]
        else:
            self.img_list[key].append(url)

    def build_html(self, path):
        html = self.header
        for key in sorted(self.img_list.keys(), key=lambda x: int(x.split("(")[0])):
            html += self.format.format(key, *self.img_list[key])
        html += self.tail
        with open(os.path.join(path, 'vis.html'), 'w') as f:
            f.write(html)
        f.close()
