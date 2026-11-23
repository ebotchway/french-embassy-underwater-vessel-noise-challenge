#!/usr/bin/env python3
"""Reproduce an explicitly NON-vessel, NON-calibrated site-context figure."""
import argparse, csv, io, zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--archive',type=Path,required=True,help='Authorized South_African_Real_Acoustics.zip')
 p.add_argument('--summary',type=Path,required=True,help='Six-row published training-only summary CSV')
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 with zipfile.ZipFile(a.archive) as z:
  stations={row['site'].split()[0]:row for row in csv.DictReader(io.StringIO(z.read('stations.csv').decode('utf-8-sig')))}
 with a.summary.open(newline='') as f: summary=list(csv.DictReader(f))
 assert len(summary)==6 and len({r['site_id'] for r in summary})==6
 fig, (ax, panel)=plt.subplots(1,2,figsize=(12,5.5),gridspec_kw={'width_ratios':[1.12,1]},layout='constrained')
 vals=[float(r['median_window_mean']) for r in summary]
 norm=colors.Normalize(vmin=min(vals),vmax=max(vals)); cmap=plt.get_cmap('viridis')
 offsets={'1':(-1.5,-.1),'2':(.32,.14),'4':(.28,.1),'5':(-1.6,.35),'7':(-3.0,-.55),'13':(.42,-.23)}
 for row in summary:
  sid=row['site_id']; s=stations[sid]
  lat,lon=float(s['latitude_candidate']),float(s['longitude_candidate'])
  val=float(row['median_window_mean']); cnt=int(row['run_entries'])
  ax.scatter(lon,lat,s=75+cnt*1.4,c=[val],cmap=cmap,norm=norm,edgecolor='black',linewidth=.65,zorder=4)
  dx,dy=offsets[sid]
  ax.annotate(row['short_name'],(lon,lat),(lon+dx,lat+dy),fontsize=8,weight='semibold',bbox={'facecolor':'white','edgecolor':'none','alpha':.74,'pad':.4},arrowprops={'arrowstyle':'-','lw':.5,'color':'#444'},zorder=5)
  if sid=='13' and s['latitude_alternative'].strip() and s['longitude_alternative'].strip():
   lat2,lon2=float(s['latitude_alternative']),float(s['longitude_alternative'])
   ax.scatter(lon2,lat2,s=62,marker='x',c='#bb2538',linewidth=1.5,zorder=6)
   ax.plot([lon,lon2],[lat,lat2],ls=':',lw=1,color='#bb2538')
 ax.set_xlim(16.5,34);ax.set_ylim(-35.2,-27)
 ax.set_xlabel('Longitude (°E), candidate site-level positions');ax.set_ylabel('Latitude (°S shown as negative)')
 ax.grid(alpha=.23);ax.xaxis.set_major_locator(MultipleLocator(3));ax.yaxis.set_major_locator(MultipleLocator(2))
 ax.set_title('A  Site-location context (not hydrophone fixes)',fontsize=11,weight='semibold')
 cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.85,pad=.01)
 cb.set_label('Median sampled run-window mean\n(source-export level; uncalibrated)',fontsize=8)
 ranked=sorted(summary,key=lambda r:float(r['median_window_mean']))
 ypos=range(len(ranked))
 panel.barh(list(ypos),[float(r['median_window_mean']) for r in ranked],color=[cmap(norm(float(r['median_window_mean']))) for r in ranked],edgecolor='#34495e',linewidth=.4)
 panel.set_yticks(list(ypos),[r['short_name'] for r in ranked],fontsize=8)
 for y,r in enumerate(ranked):panel.text(float(r['median_window_mean'])+.4,y,f"{float(r['median_window_mean']):.1f} (n={r['run_entries']})",va='center',fontsize=8)
 panel.set_xlim(0,max(vals)*1.23);panel.grid(axis='x',alpha=.2);panel.set_axisbelow(True)
 panel.set_xlabel('Median sampled run-window mean (source-export units)')
 panel.set_title('B  Descriptive total sound, NOT vessel-only',fontsize=11,weight='semibold')
 fig.suptitle('South African acoustic development stations — spatial context only',fontsize=13,weight='bold')
 fig.text(.5,-.015,'Working site candidate locations are NOT confirmed sensor deployments. Saldanha has an unresolved alternative (red ×).\nMarkers represent selected run counts; no interpolation, timestamp alignment, vessel attribution, or calibrated pressure.',ha='center',va='top',fontsize=8)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(a.output,dpi=190,bbox_inches='tight',pad_inches=.27)
 plt.close(fig)
 print('SITE CONTEXT FIGURE CREATED:',a.output)

if __name__=='__main__':main()
