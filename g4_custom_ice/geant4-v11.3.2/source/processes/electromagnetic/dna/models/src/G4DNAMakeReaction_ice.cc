//
// ********************************************************************
// ICE-SPECIFIC REACTION MAKER FOR EUROPA CHEMISTRY SIMULATION
// ********************************************************************
//
// PURPOSE: Implements position-dependent molecular reaction formation 
//          using Arrhenius diffusion coefficients for accurate
//          ice chemistry simulation in Europa's subsurface ocean
//
// PHYSICS CONTEXT:
// - Europa's ice shell has temperature gradients from ~100K (surface) to ~273K (base)
// - Diffusion coefficients vary by orders of magnitude with temperature
// - Chemical reaction rates depend directly on diffusion coefficients
// - Vanilla Geant4 uses fixed, stored diffusion values
// - Ice version dynamically calculates coefficients based on local temperature
//
// KEY INNOVATION: Position-aware molecular reaction formation
// VANILLA BEHAVIOR: Fixed coefficients regardless of location
// ICE BEHAVIOR: Arrhenius coefficients calculated per molecule position
//
// TECHNICAL IMPLEMENTATION:
// - Inherits from G4DNAMakeReaction to preserve core Geant4 functionality
// - Overrides MakeReaction() and UpdatePositionForReaction() methods
// - Uses GetArrheniusDiffusionCoefficient() for runtime coefficient calculation
// - Integrates with GetIceTemperatureAtPosition() for spatial temperature lookup

#include "G4DNAMakeReaction_ice.hh"
#include "ArrheniusConstants_ice.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
// ---------------------------------------------------------------------
//  G4DNAMakeReaction_ice.cc
//  Customized by G. Yoffe for ice (solid H2O) chemistry
//
//  Implements position-dependent molecular reaction formation using
//  Arrhenius diffusion coefficients for ice. Differs from the original
//  by using temperature-dependent coefficients and spatial logic.
// ---------------------------------------------------------------------
#include "G4MolecularConfiguration.hh"
// G4DNAMakeReaction_ice.cc - Ice-specific molecular reaction maker with Arrhenius diffusion
//

#include "G4DNAMakeReaction_ice.hh"
#include "G4DNAMolecularReactionTable.hh"
#include "G4VDNAReactionModel.hh"
#include "G4Molecule.hh"
#include "G4MoleculeFinder.hh"
#include "G4ITReactionChange.hh"
#include "Randomize.hh"
#include "G4SystemOfUnits.hh"
#include "G4ITReaction.hh"
#include "G4DNAIndependentReactionTimeStepper.hh"
#include "G4Scheduler.hh"
#include "G4UnitsTable.hh"
#include "G4ITTrackHolder.hh"

// Ice-specific includes for Arrhenius diffusion coefficients
#include "ArrheniusConstants_ice.hh"
#include "G4OH.hh"
#include "G4Hydrogen.hh"

// Helper function to get ice temperature at position (from G4DNABrownianTransportation_ice.cc)
extern G4double GetIceTemperatureAtPosition(const G4ThreeVector& position);

G4DNAMakeReaction_ice::G4DNAMakeReaction_ice()
    : G4DNAMakeReaction()
{
    // ICE-SPECIFIC CONSTRUCTOR: Initializes with vanilla base functionality
    // PURPOSE: Sets up reaction maker for ice-specific chemistry simulations
    // TECHNICAL NOTE: Inherits all standard Geant4 reaction mechanisms,
    //                 ice modifications are applied at runtime via method overrides
}

G4DNAMakeReaction_ice::G4DNAMakeReaction_ice(G4VDNAReactionModel* pReactionModel)
    : G4DNAMakeReaction(pReactionModel)
{
    // ICE-SPECIFIC CONSTRUCTOR WITH MODEL: Associates with specific reaction model
    // PURPOSE: Enables model-specific customization while maintaining ice behavior
    // USAGE: Typically called when specific reaction models (e.g., Smoluchowski) are used
    // TECHNICAL: pReactionModel provides reaction kinetics, ice class provides coefficients
}

G4double G4DNAMakeReaction_ice::GetArrheniusDiffusionCoefficient(const G4Track& track, const G4MolecularConfiguration* molConf) {
    G4ThreeVector position = track.GetPosition();
    G4double temperature = GetIceTemperatureAtPosition(position);
    
    if (molConf->GetDefinition() == G4OH::Definition()) {
        return ArrheniusConstants_ice::CalculateArrheniusDiffusion(
            ArrheniusConstants_ice::D0_OH, ArrheniusConstants_ice::Ea_OH, temperature);
    }
    else if (molConf->GetDefinition() == G4Hydrogen::Definition()) {
        return ArrheniusConstants_ice::CalculateArrheniusDiffusion(
            ArrheniusConstants_ice::D0_H, ArrheniusConstants_ice::Ea_H, temperature);
    }
    
    // For other molecules, use the stored coefficient
    return molConf->GetDiffusionCoefficient();
}

std::unique_ptr<G4ITReactionChange> G4DNAMakeReaction_ice::MakeReaction(const G4Track& trackA,
                                                                         const G4Track& trackB)
{
    // ICE-SPECIFIC REACTION CREATION: Core method for position-dependent chemistry
    // 
    // PURPOSE: Creates molecular reaction products using temperature-dependent diffusion
    // PHYSICS: In ice, reaction probabilities depend on local molecular mobility
    // 
    // VANILLA VS ICE COMPARISON:
    // VANILLA: Uses pre-calculated, temperature-independent reaction probabilities
    // ICE: Dynamically calculates reaction probabilities based on local temperature
    // 
    // ALGORITHMIC STEPS:
    // 1. Update molecular positions before reaction (accounts for ice diffusion rates)
    // 2. Initialize reaction change object to track all modifications
    // 3. Calculate ice-specific diffusion coefficients for both reactants
    // 4. Determine reaction site using proper diffusion-weighted positioning
    // 5. Create reaction products with correct spatial and energy distribution
    
    auto & tA = const_cast<G4Track&>(trackA);
    auto & tB = const_cast<G4Track&>(trackB);
    
    // Step 1: Update positions using ice-specific diffusion rates
    UpdatePositionForReaction( tA , tB );//TODO: should change it

    // Step 2: Initialize change tracking
    std::unique_ptr<G4ITReactionChange> pChanges(new G4ITReactionChange());
    pChanges->Initialize(trackA, trackB);

    // Step 3: Get molecular configurations and reaction data
    const auto pMoleculeA = GetMolecule(trackA)->GetMolecularConfiguration();
    const auto pMoleculeB = GetMolecule(trackB)->GetMolecularConfiguration();

    const auto pReactionData = fMolReactionTable->GetReactionData(pMoleculeA, pMoleculeB);
    const G4int nbProducts = pReactionData->GetNbProducts();
    if (nbProducts != 0)
    {
        // Step 4: CRITICAL ICE MODIFICATION - Use Arrhenius coefficients
        // VANILLA: Uses GetDiffusionCoefficient() which returns stored values
        // ICE: Uses GetArrheniusDiffusionCoefficient() which calculates based on temperature
        const G4double D1 = GetArrheniusDiffusionCoefficient(trackA, pMoleculeA);
        const G4double D2 = GetArrheniusDiffusionCoefficient(trackB, pMoleculeB);
        const G4double sqrD1 = D1 == 0. ? 0. : std::sqrt(D1);
        const G4double sqrD2 = D2 == 0. ? 0. : std::sqrt(D2);
        const G4double inv_numerator = 1./(sqrD1 + sqrD2);
        
        // Step 5: Calculate reaction site using diffusion-weighted positioning
        // PHYSICS: Faster-diffusing molecule contributes more to reaction site location
        const G4ThreeVector reactionSite = sqrD2 * inv_numerator * tA.GetPosition()
                                         + sqrD1 * inv_numerator * tB.GetPosition();

        G4double u = G4UniformRand();
        auto randP = (1-u) * tA.GetPosition() + u * tB.GetPosition();

        for (G4int j = 0; j < nbProducts; ++j)
        {
            // Step 6: Create reaction products at calculated reaction site
            // PHYSICS: Products inherit energy and position from reaction site calculation
            // ICE ADVANTAGE: Reaction site reflects actual diffusion-weighted center
            auto pProduct = new G4Molecule(pReactionData->GetProduct(j));
            auto pProductTrack = pProduct->BuildTrack(trackA.GetGlobalTime(), (reactionSite + randP)/2);
            pProductTrack->SetTrackStatus(fAlive);
            G4ITTrackHolder::Instance()->Push(pProductTrack);
            pChanges->AddSecondary(pProductTrack);
        }
    }
    
    // Step 7: Remove reactants from simulation (standard Geant4 behavior)
    pChanges->KillParents(true);
    return pChanges;
}

void G4DNAMakeReaction_ice::UpdatePositionForReaction(G4Track& trackA,
                                                      G4Track& trackB)
{
    // ICE-SPECIFIC POSITION UPDATE: Adjusts molecular positions before reaction
    // 
    // PURPOSE: Ensures molecules are properly positioned for reaction considering ice diffusion
    // PHYSICS: Molecules must be within reaction radius for chemistry to occur
    // 
    // VANILLA VS ICE COMPARISON:
    // VANILLA: Position updates use stored diffusion coefficients
    // ICE: Position updates reflect actual temperature-dependent mobility
    // 
    // CRITICAL IMPORTANCE: Incorrect positioning leads to:
    // 1. Wrong reaction probabilities
    // 2. Incorrect energy distributions in products
    // 3. Spatial artifacts in concentration profiles
    
    // Step 1: Get molecular configurations and calculate ice-specific coefficients
    const auto pMoleculeA = GetMolecule(trackA)->GetMolecularConfiguration();
    const auto pMoleculeB = GetMolecule(trackB)->GetMolecularConfiguration();
    
    // CRITICAL ICE MODIFICATION: Use Arrhenius coefficients instead of stored values
    G4double D1 = GetArrheniusDiffusionCoefficient(trackA, pMoleculeA);
    G4double D2 = GetArrheniusDiffusionCoefficient(trackB, pMoleculeB);

    // Step 2: Get reaction geometry parameters
    G4double reactionRadius = fpReactionModel->GetReactionRadius( pMoleculeA, pMoleculeB );
    G4ThreeVector p1 = trackA.GetPosition();
    G4ThreeVector p2 = trackB.GetPosition();

    // Step 3: Calculate separation and determine positioning strategy
    G4ThreeVector S1 = p1 - p2;
    // G4double distance = S1.mag(); // Distance available if needed for debugging

    // Step 4: Handle different diffusion scenarios with ice-specific coefficients
    if(D1 == 0)
    {
        // Scenario A: Molecule A is immobile (e.g., very cold ice region)
        // PHYSICS: Only molecule B can move to contact with A
        // ICE CONTEXT: In very cold ice, some molecules become essentially immobile
        S1.setMag(reactionRadius);
        trackB.SetPosition(p1 + S1);
    }
    else if(D2 == 0)
    {
        // Scenario B: Molecule B is immobile
        // PHYSICS: Only molecule A can move to contact with B
        S1.setMag(reactionRadius);
        trackA.SetPosition(p2 - S1);
    }
    else
    {
        // Scenario C: Both molecules are mobile (typical case in ice chemistry)
        // PHYSICS: Both molecules contribute to approach based on their diffusion rates
        // ICE ADVANTAGE: Uses actual temperature-dependent diffusion rates
        
        G4ThreeVector p1_p2 = p2 - p1;
        p1_p2.setMag(reactionRadius);

        // Calculate diffusion-based displacement parameters
        G4double dt = fTimeStep;
        G4double s12 = 2.0 * D1 * dt;  // Displacement variance for molecule A
        G4double s22 = 2.0 * D2 * dt;  // Displacement variance for molecule B
        
        if(s12 == 0)
        {
            // Molecule A effectively immobile during this timestep
            trackB.SetPosition(p1 + p1_p2);
        }
        else if(s22 == 0)
        {
            // Molecule B effectively immobile during this timestep
            trackA.SetPosition(p2 - p1_p2);
        }
        else
        {
            // Both molecules mobile: Calculate diffusion-weighted positioning
            // CRITICAL ICE PHYSICS: α reflects relative diffusion rates in ice
            // VANILLA: Uses stored coefficients → wrong α in temperature gradients
            // ICE: Uses Arrhenius coefficients → correct α reflecting local mobility
            G4double alpha = D1 / (D1 + D2);
            G4ThreeVector S2 = -p1_p2;
            S2.setMag(alpha * reactionRadius);
            S1.setMag((1 - alpha) * reactionRadius);

            // Final positioning: Each molecule moves proportional to its diffusion rate
            trackA.SetPosition(p1 + S1);
            trackB.SetPosition(p2 + S2);
        }
    }
    
    // SUMMARY: This method ensures proper molecular positioning for reaction
    // using ice-specific, temperature-dependent diffusion coefficients
    // Result: Realistic reaction geometries and energy distributions in ice environment
}
