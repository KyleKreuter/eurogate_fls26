export type Region = "all" | "north-sea" | "mediterranean" | "atlantic";

export interface TerminalLocation {
  id: string;
  city: string;
  country: string;
  countryCode: string;
  coordinates: [number, number]; // [lng, lat]
  active: boolean;
  region: Exclude<Region, "all">;
}

/** All 9 Eurogate terminal locations with real geographic coordinates */
export const TERMINALS: TerminalLocation[] = [
  // Germany — North Sea
  { id: "bremerhaven", city: "Bremerhaven", country: "Germany", countryCode: "DE", coordinates: [8.58, 53.55], active: false, region: "north-sea" },
  { id: "hamburg", city: "Hamburg", country: "Germany", countryCode: "DE", coordinates: [9.97, 53.55], active: true, region: "north-sea" },
  { id: "wilhelmshaven", city: "Wilhelmshaven", country: "Germany", countryCode: "DE", coordinates: [8.13, 53.51], active: false, region: "north-sea" },
  // Italy — Mediterranean
  { id: "la-spezia", city: "La Spezia", country: "Italy", countryCode: "IT", coordinates: [9.82, 44.10], active: false, region: "mediterranean" },
  { id: "ravenna", city: "Ravenna", country: "Italy", countryCode: "IT", coordinates: [12.20, 44.42], active: false, region: "mediterranean" },
  { id: "salerno", city: "Salerno", country: "Italy", countryCode: "IT", coordinates: [14.77, 40.68], active: false, region: "mediterranean" },
  // Morocco — Atlantic
  { id: "tanger", city: "Tanger", country: "Morocco", countryCode: "MA", coordinates: [-5.80, 35.77], active: false, region: "atlantic" },
  // Cyprus — Mediterranean
  { id: "limassol", city: "Limassol", country: "Cyprus", countryCode: "CY", coordinates: [33.04, 34.68], active: false, region: "mediterranean" },
  // Egypt — Mediterranean
  { id: "damietta", city: "Damietta", country: "Egypt", countryCode: "EG", coordinates: [31.82, 31.42], active: false, region: "mediterranean" },
];

export const COUNTRY_GROUPS = [
  { code: "DE", label: "Deutschland" },
  { code: "IT", label: "Italia" },
  { code: "MA", label: "Maroc" },
  { code: "CY", label: "Cyprus" },
  { code: "EG", label: "Egypt" },
] as const;

export const TERMINAL_STATS = [
  { value: "12.5M", label: "Annual TEU Capacity" },
  { value: "5,000+", label: "Logistics Specialists" },
  { value: "18.0M", label: "Max. Berth Depth" },
];

export const REGIONS: { id: Region; label: string; iconName: string }[] = [
  { id: "all", label: "Global Network", iconName: "globe" },
  { id: "north-sea", label: "North Sea", iconName: "anchor" },
  { id: "mediterranean", label: "Mediterranean", iconName: "sun" },
  { id: "atlantic", label: "Atlantic", iconName: "waves" },
];
