import type { LineSeries } from "@nivo/line";
import type { BarDatum } from "@nivo/bar";

// Stündliche Lastprognose (kW) über 24h
export const loadForecastData: LineSeries[] = [
  {
    id: "Prognose",
    data: [
      { x: "00:00", y: 320 },
      { x: "01:00", y: 305 },
      { x: "02:00", y: 290 },
      { x: "03:00", y: 280 },
      { x: "04:00", y: 275 },
      { x: "05:00", y: 285 },
      { x: "06:00", y: 310 },
      { x: "07:00", y: 365 },
      { x: "08:00", y: 420 },
      { x: "09:00", y: 475 },
      { x: "10:00", y: 510 },
      { x: "11:00", y: 540 },
      { x: "12:00", y: 555 },
      { x: "13:00", y: 560 },
      { x: "14:00", y: 545 },
      { x: "15:00", y: 520 },
      { x: "16:00", y: 490 },
      { x: "17:00", y: 455 },
      { x: "18:00", y: 420 },
      { x: "19:00", y: 390 },
      { x: "20:00", y: 370 },
      { x: "21:00", y: 355 },
      { x: "22:00", y: 340 },
      { x: "23:00", y: 325 },
    ],
  },
  {
    id: "Ist-Werte",
    data: [
      { x: "00:00", y: 315 },
      { x: "01:00", y: 298 },
      { x: "02:00", y: 285 },
      { x: "03:00", y: 278 },
      { x: "04:00", y: 270 },
      { x: "05:00", y: 290 },
      { x: "06:00", y: 325 },
      { x: "07:00", y: 380 },
      { x: "08:00", y: 435 },
      { x: "09:00", y: 490 },
      { x: "10:00", y: 520 },
      { x: "11:00", y: 548 },
      { x: "12:00", y: 562 },
      { x: "13:00", y: 570 },
      { x: "14:00", y: 550 },
      { x: "15:00", y: 515 },
      { x: "16:00", y: 480 },
      { x: "17:00", y: 445 },
      { x: "18:00", y: 410 },
      { x: "19:00", y: 385 },
      { x: "20:00", y: 365 },
      { x: "21:00", y: 348 },
      { x: "22:00", y: 335 },
      { x: "23:00", y: 320 },
    ],
  },
];

// Wochentags-Durchschnitt (kW)
export const weeklyAvgData: BarDatum[] = [
  { day: "Mo", avg: 425 },
  { day: "Di", avg: 440 },
  { day: "Mi", avg: 455 },
  { day: "Do", avg: 460 },
  { day: "Fr", avg: 435 },
  { day: "Sa", avg: 380 },
  { day: "So", avg: 350 },
];

// KPI-Zusammenfassung
export const kpiData = {
  peakLoad: 570,
  avgLoad: 405,
  minLoad: 270,
  temperature: 8.3,
  windSpeed: 18.5,
  reeferCount: 142,
};
