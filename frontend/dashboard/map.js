document.addEventListener('DOMContentLoaded', () => {
    // Initialize map encompassing Europe/North Africa
    const map = L.map('eurogate-map', {
        zoomControl: false 
    }).setView([43.0, 15.0], 5);

    L.control.zoom({
        position: 'topright'
    }).addTo(map);

    // Ultra-clean Corporate Map Engine using CartoDB Positron (No Labels for Maximum Simplicity)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap &copy; CARTO'
    }).addTo(map);

    // Baseline Blue Marker Base
    const defaultIcon = L.icon({
        iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
        iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
        shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        popupAnchor: [1, -34],
        shadowSize: [41, 41]
    });
    
    // Highlighted Orange Marker specifically for Hamburg interaction
    const hamburgIcon = L.icon({
        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-orange.png',
        shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        popupAnchor: [1, -34],
        shadowSize: [41, 41]
    });

    // 9 Exact Geocoordinates for Eurogate Target Locations
    const harbours = [
        { name: "Bremerhaven", coords: [53.5396, 8.5809], redirect: null },
        { name: "Hamburg", coords: [53.5511, 9.9937], redirect: "hamburg.html" },
        { name: "Wilhelmshaven", coords: [53.5215, 8.1118], redirect: null },
        { name: "La Spezia", coords: [44.1025, 9.8241], redirect: null },
        { name: "Ravenna", coords: [44.4184, 12.1997], redirect: null },
        { name: "Salerno", coords: [40.6824, 14.7681], redirect: null },
        { name: "Tanger", coords: [35.8890, -5.5015], redirect: null }, 
        { name: "Limassol", coords: [34.6749, 33.0384], redirect: null },
        { name: "Damietta", coords: [31.4165, 31.8133], redirect: null }
    ];

    harbours.forEach(port => {
        // Render orange solely on interactable Hamburg port
        const markerIcon = port.name === 'Hamburg' ? hamburgIcon : defaultIcon;
        const marker = L.marker(port.coords, { 
            icon: markerIcon,
            title: port.name
        }).addTo(map);

        // Hover tooltip binding
        marker.bindTooltip(port.name === 'Hamburg' ? 'Hamburg (Click For Dashboard)' : port.name, {
            permanent: false, 
            direction: 'top',
            className: 'eurogate-tooltip',
            offset: [0, -35]
        });

        // Click routing logic
        if (port.redirect) {
            marker.on('click', () => {
                window.location.href = port.redirect;
            });
            // Force cursor to standard hand
            marker.on('mouseover', () => {
                const el = marker.getElement();
                if(el) el.classList.add('clickable-marker');
            });
        }
    });
});
